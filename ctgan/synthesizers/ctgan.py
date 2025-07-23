"""CTGAN module."""

import warnings

import numpy as np
import math
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.errors import InvalidDataError
from ctgan.synthesizers.base import BaseSynthesizer, random_state


class Discriminator(Module):
    """Discriminator CTGAN – thêm nhánh xử lý label song song layer đầu."""

    def __init__(self, input_dim, discriminator_dim, label_dim, pac=10):
        """
        input_dim        : data_dim + cond_dim   (một sample, chưa nhân pac)
        discriminator_dim: tuple hidden sizes    (ví dụ (256, 256))
        label_dim        : chiều one-hot của nhãn
        pac              : pack size CTGAN gốc
        """
        super(Discriminator, self).__init__()

        self.pac = pac
        self.pacdim_main  = input_dim * pac          # phần (data+cond)
        self.pacdim_label = label_dim * pac          # phần label
        first_hidden = discriminator_dim[0]

        # ── Nhánh MAIN (data+cond) – giữ nguyên cấu trúc cũ ────────────────
        self.main_head = Sequential(
            Linear(self.pacdim_main, first_hidden),
            LeakyReLU(0.2),
            Dropout(0.5),
        )

        # ── Nhánh LABEL – 1 layer, hidden = 2×first_hidden ────────────────
        self.label_head = Sequential(
            Linear(self.pacdim_label, first_hidden * 2),
            LeakyReLU(0.2),
            Dropout(0.5),
        )

        # ── Tail sau khi concat hai nhánh ─────────────────────────────────
        dim = first_hidden + first_hidden * 2
        tail_layers = []
        for h in list(discriminator_dim)[1:]:
            tail_layers += [Linear(dim, h), LeakyReLU(0.2), Dropout(0.5)]
            dim = h
        tail_layers.append(Linear(dim, 1))
        self.tail = Sequential(*tail_layers)

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP) – thêm label nhưng GIỮ nguyên var name
    # ------------------------------------------------------------------
    def calc_gradient_penalty(
        self, real_main, real_label, fake_main, fake_label, device='cpu', lambda_=10
    ):
        """real_* / fake_* đã là tensor chưa pack."""
        assert real_main.size(0) % self.pac == 0, 'batch phải chia hết pac'

        real = torch.cat([real_main, real_label], dim=1)
        fake = torch.cat([fake_main, fake_label], dim=1)

        alpha = torch.rand(real.size(0) // self.pac, 1, 1, device=device)
        alpha = alpha.repeat(1, self.pac, real.size(1)).reshape(real.size())

        interpolates = alpha * real + (1 - alpha) * fake
        inter_main, inter_label = torch.split(
            interpolates, [real_main.size(1), real_label.size(1)], dim=1
        )
        disc_interpolates = self.forward(inter_main, inter_label)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(real.size(0), -1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        return penalty

    # ------------------------------------------------------------------
    # Forward – truyền RIÊNG main_input (data+cond) & label_input
    # ------------------------------------------------------------------
    def forward(self, main_input, label_input):
        """main_input: (B, data+cond)  |  label_input: (B, label_dim)"""
        assert main_input.size(0) % self.pac == 0, 'batch phải chia hết pac'

        b_pack = main_input.size(0) // self.pac
        h_main  = self.main_head(main_input.reshape(b_pack, -1))
        h_label = self.label_head(label_input.reshape(b_pack, -1))

        features = torch.cat([h_main, h_label], dim=1)
        return self.tail(features)


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for CTGAN – xử lý nhánh (noise ⊕ cond) và nhánh label song song."""

    def __init__(self, embedding_dim, generator_dim, data_dim, label_dim):
        """
        Parameters
        ----------
        embedding_dim : int
            Kích thước noise ⊕ cond_vec (main branch input).
        generator_dim : tuple[int]
            Hidden sizes cho các Residual; độ dài ≥ 1.
        data_dim : int
            Số chiều đầu ra (≡ tổng dim của transformer).
        label_dim : int
            Số chiều one-hot của nhãn.
        """
        super().__init__()

        # ── Branch MAIN (noise + cond) – Residual đầu giữ nguyên ──────────
        self.seq_main = Sequential(Residual(embedding_dim, generator_dim[0]))
        dim_main_out = embedding_dim + generator_dim[0]

        # ── Branch LABEL – Linear → ReLU, hidden = 2 × first_hidden ───────
        self.seq_label = Sequential(
            Linear(label_dim, generator_dim[0] * 2),
            ReLU(),
        )
        dim_label_out = generator_dim[0] * 2

        # ── Tail - concat + các Residual còn lại + Linear cuối ────────────
        dim_total = dim_main_out + dim_label_out
        tail = []
        for h in generator_dim[1:]:
            tail.append(Residual(dim_total, h))
            dim_total += h
        tail.append(Linear(dim_total, data_dim))
        self.seq_tail = Sequential(*tail)

    # ------------------------------------------------------------------ #
    # forward(main_input, label_input) – KHÔNG còn tách trong hàm        #
    # ------------------------------------------------------------------ #
    def forward(self, main_input, label_input):
        """
        Parameters
        ----------
        main_input  : Tensor (B, embedding_dim)
            noise ⊕ cond_vec.
        label_input : Tensor (B, label_dim)
            one-hot label.

        Returns
        -------
        Tensor (B, data_dim)
            Đầu ra đã qua các activation nội bộ.
        """
        h_main  = self.seq_main(main_input)
        h_label = self.seq_label(label_input)
        h = torch.cat([h_main, h_label], dim=1)
        return self.seq_tail(h)


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        cond_loss_weight = 1.0
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._cond_loss_weight = cond_loss_weight

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _validate_null_data(self, train_data, discrete_columns):
        """Check whether null values exist in continuous ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise InvalidDataError(
                'CTGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None, class_column=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns, class_column=class_column)

        train_data, label_onehot = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, label_onehot, self._transformer.output_info_list, self._log_frequency
        )

        cond_dim   = self._data_sampler.dim_cond_vec()
        label_dim  = self._data_sampler.dim_label_vec()
        data_dim = self._transformer.output_dimensions

        # ---------- Generator -------------------------------------------- #
        self._generator = Generator(
            self._embedding_dim + cond_dim,   ### FIX
            self._generator_dim,
            data_dim,
            label_dim                                    # param mới của class Generator
        ).to(self._device)

        # ---------- Discriminator ---------------------------------------- #
        discriminator = Discriminator(
            data_dim + cond_dim,             ### FIX
            self._discriminator_dim,
            label_dim,                                   # param mới của class Discriminator
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    
                    # sample_condvec giờ trả 5 phần tử
                    cond, mask, col, opt, label_batch = self._data_sampler.sample_condvec(self._batch_size)
                    l1 = torch.from_numpy(label_batch).to(self._device)     

                    if cond is None:
                        c1, m1, col, opt = None, None, None, None
                        fakez_main = fakez                                   # chỉ noise
                        real_x, real_l = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                        l2 = torch.from_numpy(real_l).to(self._device)
                        real_main = torch.from_numpy(real_x.astype('float32')).to(self._device)
                    else:
                        c1 = torch.from_numpy(cond).to(self._device)
                        m1 = torch.from_numpy(mask).to(self._device)
                        fakez_main = torch.cat([fakez, c1], dim=1)          # noise+cond
                        perm = np.random.permutation(self._batch_size)  # dùng chung
                        real_x, real_l = self._data_sampler.sample_data(
                            train_data, self._batch_size,
                            col[perm], opt[perm]
                        )
                        c2 = c1[perm]           # permute cond
                        l2 = torch.from_numpy(real_l).to(self._device)
                        real_main = torch.cat([torch.from_numpy(real_x.astype('float32')).to(self._device),
                                            c2], dim=1)

                    # ---- Generator forward -------------------------------------
                    fake = self._generator(fakez_main, l1)                  # (main, label)
                    fake_act = self._apply_activate(fake)
                    fake_main = torch.cat([fake_act, c1], dim=1) if c1 is not None else fake_act

                    # ---- Discriminator forward ---------------------------------
                    y_fake = discriminator(fake_main, l1)
                    y_real = discriminator(real_main, l2)

                    pen = discriminator.calc_gradient_penalty(
                        real_main, l2, fake_main, l1, self._device
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # ----------------  GENERATOR update  ----------------------------
                fakez = torch.normal(mean=mean, std=std)
                cond, mask, col, opt, label_batch = self._data_sampler.sample_condvec(
                    self._batch_size
                )
                l1 = torch.from_numpy(label_batch).to(self._device)

                if cond is None:
                    c1, m1 = None, None
                    fakez_main = fakez
                else:
                    c1 = torch.from_numpy(cond).to(self._device)
                    m1 = torch.from_numpy(mask).to(self._device)
                    fakez_main = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez_main, l1)
                fake_act = self._apply_activate(fake)
                fake_main = torch.cat([fake_act, c1], dim=1) if c1 is not None else fake_act
                y_fake = discriminator(fake_main, l1)

                cross_entropy = 0 if cond is None else self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy * self._cond_loss_weight

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
    '''
    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy*self._cond_loss_weight

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
                
                '''
    
    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample synthetic data (optionally conditioned by one discrete value)."""

        # ----- 1. Chuẩn bị condvec toàn cục (nếu user yêu cầu) ---------------
        if condition_column is not None and condition_value is not None:
            cond_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condvec = self._data_sampler.generate_cond_from_condition_column_info(
                cond_info, self._batch_size
            )
        else:
            global_condvec = None

        # ----- 2. Lặp batch ---------------------------------------------------
        steps = n // self._batch_size + 1
        data, label_all = [], []
        for _ in range(steps):
            # -- noise ---------------------------------------------------------
            mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
            std  = mean + 1
            z    = torch.normal(mean=mean, std=std)

            # -- condvec & label_vec ------------------------------------------
            if global_condvec is not None:
                condvec = global_condvec.copy()
                # lấy label ngẫu nhiên từ sampler
                _, _, _, _, label_batch = self._data_sampler.sample_condvec(self._batch_size)
            else:
                condvec, _, _, _, label_batch = self._data_sampler.sample_condvec(
                    self._batch_size
                )

            # condvec -> tensor (có thể None)
            if condvec is not None:
                c1 = torch.from_numpy(condvec).to(self._device)
                fakez_main = torch.cat([z, c1], dim=1)
            else:
                c1 = None
                fakez_main = z

            # label tensor
            l1 = torch.from_numpy(label_batch).to(self._device)

            # -- Generator -----------------------------------------------------
            fake = self._generator(fakez_main, l1)
            fake_act = self._apply_activate(fake)

            data.append(fake_act.detach().cpu().numpy())
            label_all.append(label_batch)

        # ----- 3. Gộp & cắt đúng n -------------------------------------------
        data = np.concatenate(data, axis=0)[:n]
        label_all = np.concatenate(label_all, axis=0)[:n]

        # ----- 4. Giải mã về không gian gốc ----------------------------------
        if getattr(self._transformer, "_class_column", None) is not None:
            return self._transformer.inverse_transform(data, y_cond=label_all)
        else:
            return self._transformer.inverse_transform(data)
        
    @random_state
    def sample_exact(self, n, condition_column, condition_value, max_tries=30):
        """
        Generate *exactly* ``n`` rows that satisfy
        ``df[condition_column] == condition_value``.

        The method keeps sampling full batches conditioned on the requested
        category until it has gathered at least ``n`` matching rows or the
        number of attempts exceeds ``max_tries``.

        Args
        ----
        n (int)
            Exact number of rows to return.
        condition_column (str)
            Name of the discrete column to condition on.
        condition_value (str)
            Target category in ``condition_column`` that every returned row must contain.
        batch_size (int, optional)
            Overrides the modelʼs default ``self._batch_size`` for this call.
        max_tries (int, default = 30)
            Maximum number of full-batch sampling attempts before giving up.

        Raises
        ------
        ValueError
            If after ``max_tries`` attempts the method still cannot collect
            ``n`` valid rows.

        Returns
        -------
        pandas.DataFrame
            Exactly ``n`` synthetic rows, each with
            ``condition_column == condition_value``.
        """
        batch_size = self._batch_size
        condition_info = self._transformer.convert_column_name_value_to_id(
            condition_column, condition_value
        )
        global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
            condition_info, batch_size
        )

        sample = []
        rows_collected = 0
        tries = 0
        
        # Ceiling division: minimum number of batches we *might* need
        target_batches = math.ceil(n / batch_size)

        while rows_collected < n and tries < max_tries:
            tries += 1

            # --------- sample one conditioned batch ---------
            z = torch.normal(
                mean=torch.zeros(batch_size, self._embedding_dim, device=self._device),
                std=1.0
            )
            z = torch.cat([z, torch.from_numpy(global_condition_vec).to(self._device)], dim=1)
            fake = self._generator(z)
            fakeact = self._apply_activate(fake).detach().cpu().numpy()
            real_batch = self._transformer.inverse_transform(fakeact)

            # --------- keep only rows that meet the condition ---------
            real_batch = real_batch[real_batch[condition_column] == condition_value]

            sample.append(real_batch)
            rows_collected += len(real_batch)

            # Optional early concat to avoid many tiny DataFrames
            if len(sample) >= target_batches:  # avoid concat too many times
                sample = [pd.concat(sample, ignore_index=True)]
                rows_collected = len(sample[0])
        
        result = pd.concat(sample, ignore_index=True)
        if len(result) < n:
            raise ValueError(
                f'Only generated {len(result)} rows after {tries} tries '
                f'(requested {n}). Try increasing max_tries or batch_size.'
            )

        return result.iloc[:n], tries

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
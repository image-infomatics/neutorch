import math
import torch
import torch.nn as nn


class ConvPass(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            activation,
            padding='valid'):

        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)

        layers = []

        for kernel_size in kernel_sizes:

            self.dims = len(kernel_size)

            conv = {
                2: torch.nn.Conv2d,
                3: torch.nn.Conv3d,
                4: Conv4d
            }[self.dims]

            if padding == 'same':
                pad = tuple(k//2 for k in kernel_size)
            else:
                pad = 0

            try:
                layers.append(
                    conv(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=pad))
            except KeyError:
                raise RuntimeError(
                    "%dD convolution not implemented" % self.dims)

            in_channels = out_channels

            if activation is not None:
                layers.append(activation())

        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_pass(x)
        return x


class Downsample(torch.nn.Module):

    def __init__(
            self,
            downsample_factor):

        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(
            downsample_factor,
            stride=downsample_factor)

    def forward(self, x):

        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d" % (
                        x.size(),
                        self.downsample_factor,
                        self.dims - d))

        return self.down(x)


class Upsample(torch.nn.Module):

    def __init__(
            self,
            scale_factor,
            mode='transposed_conv',
            in_channels=None,
            out_channels=None,
            crop_factor=None,
            next_conv_kernel_sizes=None):

        super(Upsample, self).__init__()

        assert (crop_factor is None) == (next_conv_kernel_sizes is None), \
            "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes

        self.dims = len(scale_factor)

        if mode == 'transposed_conv':

            up = {
                2: torch.nn.ConvTranspose2d,
                3: torch.nn.ConvTranspose3d
            }[self.dims]

            self.up = up(
                in_channels,
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor)

        else:

            self.up = torch.nn.Upsample(
                scale_factor=scale_factor,
                mode=mode)

    def crop_to_factor(self, x, factor, kernel_sizes):
        '''Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        '''

        shape = x.size()
        spatial_shape = shape[-self.dims:]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes)
            for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c)/f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n*f + c
            for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:

            if not all((
                (t > c) for t, c in zip(
                    target_spatial_shape,
                    convolution_crop))
            ):
                print(
                    f"Feature map with shape {shape} is too small to ensure translation equivariance with factor {factor} and following convolutions {kernel_sizes}")

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):

        g_up = self.up(g_out)

        # if self.next_conv_kernel_sizes is not None:
        #     g_cropped = self.crop_to_factor(
        #         g_up,
        #         self.crop_factor,
        #         self.next_conv_kernel_sizes)
        # else:
        #     g_cropped = g_up
        g_cropped = g_up
        f_cropped = self.crop(f_left, g_cropped.size()[-self.dims:])
        x = torch.cat([f_cropped, g_cropped], dim=1)

        return x


class UNet(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            activation='ReLU',
            fov=(1, 1, 1),
            voxel_size=(1, 1, 1),
            num_fmaps_out=None,
            num_heads=1,
            constant_upsample=False,
            padding='valid'):
        '''Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        This U-Net performs only "valid" convolutions, i.e., sizes of the
        feature maps decrease after each convolution. It will perfrom 4D
        convolutions as long as ``length`` is greater than 1. As soon as
        ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            in_channels:

                The number of input channels.

            num_fmaps:

                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels``
                dimension.

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            fov (optional):

                Initial field of view in physical units

            voxel_size (optional):

                Size of a voxel in the input data, in physical units

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            padding (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).
        '''

        super(UNet, self).__init__()

        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps

        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3, 3, 3), (3, 3, 3)]]*self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3, 3, 3), (3, 3, 3)]]*(self.num_levels - 1)

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(
                    f*ff
                    for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList([
            ConvPass(
                in_channels
                if level == 0
                else num_fmaps*fmap_inc_factor**(level - 1),
                num_fmaps*fmap_inc_factor**level,
                kernel_size_down[level],
                activation=activation,
                padding=padding)
            for level in range(self.num_levels)
        ])
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList([
            Downsample(downsample_factors[level])
            for level in range(self.num_levels - 1)
        ])

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList([
            nn.ModuleList([
                Upsample(
                    downsample_factors[level],
                    mode='nearest' if constant_upsample else 'transposed_conv',
                    in_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    out_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    crop_factor=crop_factors[level],
                    next_conv_kernel_sizes=kernel_size_up[level])
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

        # right convolutional passes
        self.r_conv = nn.ModuleList([
            nn.ModuleList([
                ConvPass(
                    num_fmaps*fmap_inc_factor**level +
                    num_fmaps*fmap_inc_factor**(level + 1),
                    num_fmaps*fmap_inc_factor**level
                    if num_fmaps_out is None or level != 0
                    else num_fmaps_out,
                    kernel_size_up[level],
                    activation=activation,
                    padding=padding)
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:

            fs_out = [f_left]*self.num_heads

        else:

            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            # up, concat, and crop
            fs_right = [
                self.r_up[h][i](f_left, gs_out[h])
                for h in range(self.num_heads)
            ]

            # convolve
            fs_out = [
                self.r_conv[h][i](fs_right[h])
                for h in range(self.num_heads)
            ]

        return fs_out

    def forward(self, x):

        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        return y


class Conv4d(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True,
            bias_initializer=None,
            kernel_initializer=None):
        '''
        Performs a 4D convolution of the ``(t, z, y, x)`` dimensions of a
        tensor with shape ``(b, c, l, d, h, w)`` with ``k`` filters. The output
        tensor will be of shape ``(b, k, l', d', h', w')``. ``(l', d', h',
        w')`` will be smaller than ``(l, d, h, w)`` if a padding smaller than
        half of the kernel size was chosen.

        Args:

            in_channels (int):

                Number of channels in the input image.

            out_channels (int):

                Number of channels produced by the convolution.

            kernel_size (int or tuple):

                Size of the convolving kernel.

            stride (int or tuple, optional):

                Stride of the convolution. Default: 1

            padding (int or tuple, optional):

                Zero-padding added to all four sides of the input. Default: 0

            padding_mode (string, optional).

                Accepted values `zeros` and `circular`. Default: `zeros`

            dilation (int or tuple, optional):

                Spacing between kernel elements. Default: 1

            groups (int, optional):

                Number of blocked connections from input channels to output
                channels. Default: 1

            bias (bool, optional):

                If ``True``, adds a learnable bias to the output. Default:
                ``True``

            bias_initializer, kernel_initializer (callable):

                An optional initializer for the bias and the kernel weights.

        This operator realizes a 4D convolution by performing several 3D
        convolutions. The following example demonstrates how this works for a
        2D convolution as a sequence of 1D convolutions::

            I.shape == (h, w)
            k.shape == (U, V) and U%2 = V%2 = 1

            # we assume kernel is indexed as follows:
            u in [-U/2,...,U/2]
            v in [-V/2,...,V/2]

            (k*I)[i,j] = Σ_u Σ_v k[u,v] I[i+u,j+v]
                       = Σ_u (k[u]*I[i+u])[j]
            (k*I)[i]   = Σ_u k[u]*I[i+u]
            (k*I)      = Σ_u k[u]*I_u, with I_u[i] = I[i+u] shifted I by u

            Example:

                I = [
                    [0,0,0],
                    [1,1,1],
                    [1,1,0],
                    [1,0,0],
                    [0,0,1]
                ]

                k = [
                    [1,1,1],
                    [1,2,1],
                    [1,1,3]
                ]

                # convolve every row in I with every row in k, comments show
                # output row the convolution contributes to
                (I*k[0]) = [
                    [0,0,0], # I[0] with k[0] ⇒ (k*I)[ 1] ✔
                    [2,3,2], # I[1] with k[0] ⇒ (k*I)[ 2] ✔
                    [2,2,1], # I[2] with k[0] ⇒ (k*I)[ 3] ✔
                    [1,1,0], # I[3] with k[0] ⇒ (k*I)[ 4] ✔
                    [0,1,1]  # I[4] with k[0] ⇒ (k*I)[ 5]
                ]
                (I*k[1]) = [
                    [0,0,0], # I[0] with k[1] ⇒ (k*I)[ 0] ✔
                    [3,4,3], # I[1] with k[1] ⇒ (k*I)[ 1] ✔
                    [3,3,1], # I[2] with k[1] ⇒ (k*I)[ 2] ✔
                    [2,1,0], # I[3] with k[1] ⇒ (k*I)[ 3] ✔
                    [0,1,2]  # I[4] with k[1] ⇒ (k*I)[ 4] ✔
                ]
                (I*k[2]) = [
                    [0,0,0], # I[0] with k[2] ⇒ (k*I)[-1]
                    [4,5,2], # I[1] with k[2] ⇒ (k*I)[ 0] ✔
                    [4,2,1], # I[2] with k[2] ⇒ (k*I)[ 1] ✔
                    [1,1,0], # I[3] with k[2] ⇒ (k*I)[ 2] ✔
                    [0,3,1]  # I[4] with k[2] ⇒ (k*I)[ 3] ✔
                ]

                # the sum of all valid output rows gives k*I (here shown for
                # row 2)
                (k*I)[2] = (
                    [2,3,2] +
                    [3,3,1] +
                    [1,1,0] +
                ) = [6,7,3]
        '''

        super(Conv4d, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert stride == 1, \
            'Strides other than 1 not yet implemented!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(l_k):

            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=(d_k, h_k, w_k),
                                           padding=self.padding)

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv3d_layer.weight)
            if self.bias_initializer is not None:
                self.bias_initializer(conv3d_layer.bias)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        l_o = l_i + 2 * self.padding - l_k + 1

        # Output tensors for each 3D frame
        frame_results = l_o * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):

            for j in range(l_i):

                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue

                frame_conv3d = \
                    self.conv3d_layers[i](input[:, :, j, :]
                                          .view(b, c_i, d_i, h_i, w_i))

                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d

        return torch.stack(frame_results, dim=2)


class Convolve(torch.nn.Module):

    def __init__(
            self,
            model,
            in_channels,
            out_channels,
            kernel_size=(1, 1, 1)):

        super().__init__()

        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        conv = torch.nn.Conv3d

        self.conv_pass = torch.nn.Sequential(
            conv(
                self.in_channels,
                self.out_channels,
                self.kernel_size),
            torch.nn.Sigmoid())

    def forward(self, x):

        y = self.model.forward(x)

        return self.conv_pass(y)


class FunUnet(nn.Module):
    """
    Residual Symmetric U-Net with down/upsampling in/output.

    num_fmaps:

        The number of feature maps in the first layer. This is also the
        number of output feature maps. Stored in the ``channels``
        dimension.

    fmap_inc_factor:

        By how much to multiply the number of feature maps between
        layers. If layer 0 has ``k`` feature maps, layer ``l`` will
        have ``k*fmap_inc_factor**l``.

    downsample_factors:

        List of tuples ``(z, y, x)`` to use to down- and up-sample the
        feature maps between layers.

    """

    def __init__(self, in_channels: int, out_channels: int, num_fmaps: int = 12, fmap_inc_factors: int = 6, downsample_factors=[(1, 2, 2), (2, 4, 4), (1, 4, 4,)]):
        super().__init__()

        self.unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors,
            padding='same')

        kernel_size = (1, 1, 1)
        self.conv = torch.nn.Conv3d(num_fmaps, out_channels, kernel_size)

    def forward(self, x):

        x = self.unet(x)
        x = self.conv(x)

        return x

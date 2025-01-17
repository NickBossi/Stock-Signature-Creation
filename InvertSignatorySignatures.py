import torch
import sys
from typing import Optional

def signature_channels(input_channel_size: int, depth: int, scalar_term: bool = False) -> int:
        if (input_channel_size < 1):
            print("input_channels must be at least 1")
        if (depth < 1):
            print("depth must be at least 1")

        if (input_channel_size == 1):
            if (scalar_term):
                return depth + 1
            else:
                return depth
        else:
            # In theory it'd probably be slightly quicker to calculate this via the geometric formula, but that
            # involves a division which gives inaccurate results for large numbers.
            output_channels = input_channel_size
            mul_limit = sys.maxsize / input_channel_size
            add_limit = sys.maxsize - input_channel_size
            for i in range(1,depth): # or depth+1, could also be 0,depth (int64_t depth_index = 1; depth_index < depth; ++depth_index) {
                if (output_channels > mul_limit):
                    print("Integer overflow detected.")
                output_channels *= input_channel_size
                if (output_channels > add_limit):
                    print("Integer overflow detected.")
                output_channels += input_channel_size

            if (scalar_term):
                return output_channels + 1
            else:
                return output_channels

def extract_signature_term(sigtensor: torch.Tensor, channels: int, depth: int,
                           scalar_term: bool = False) -> torch.Tensor:
    r"""Extracts a particular term from a signature.

    The signature to depth :math:`d` of a batch of paths in :math:`\mathbb{R}^\text{C}` is a tensor with
    :math:`C + C^2 + \cdots + C^d` channels. (See :func:`signatory.signature`.) This function extracts the :attr:`depth`
    term of that, returning a tensor with just :math:`C^\text{depth}` channels.

    Arguments:
        sigtensor (:class:`torch.Tensor`): The signature to extract the term from. Should be a result from the
            :func:`signatory.signature` function.

        channels (int): The number of input channels :math:`C`.

        depth (int): The depth of the term to be extracted from the signature.

        scalar_term (bool, optional): Whether the signature was called with scalar_term=True or not.

    Returns:
        The :class:`torch.Tensor` corresponding to the :attr:`depth` term of the signature.
    """

    if channels < 1:
        raise ValueError("in_channels must be at least 1")

    if depth == 1:
        start = int(scalar_term)
    else:
        start = signature_channels(channels, depth - 1, scalar_term)
    return sigtensor.narrow(dim=-1, start=start, length=channels ** depth)

def get_insertion_matrix(signature, insertion_position, depth, channels):
    """This function creates the matrix corresponding to the insertion map, used in the optimization problem.

     Arguments:
        signature (:class:`torch.Tensor`): As :func:`signatory.invert_signature`.

        insertion_position (int): Insertion spot.

        depth (int): As :func:`signatory.invert_signature`.

        channels (int): As :func:`signatory.invert_signature`.

    Returns:
        The :class:`torch.Tensor` representing the linear insertion map, of shape
        :math:`(N, \text{channels}, \text{channels}^{(\text{depth} +1)}', where :math:`N' is the batch size.

    """

    batch = signature.shape[0]
    B = torch.cat(batch * [torch.eye(channels)])
    new_shape = [batch] + [channels] + [1] * (insertion_position - 1) + [channels] + [1] * (depth + 1 -
                                                                                            insertion_position)
    repeat_points = [1, 1] + [channels] * (insertion_position - 1) + [1] + [channels] * (depth + 1 - insertion_position)
    new_B = B.view(new_shape)
    new_B = new_B.repeat(repeat_points)

    last_signature_term = extract_signature_term(signature, channels, depth)
    last_signature_term = last_signature_term.view([batch] + [channels] * int(depth)).unsqueeze(insertion_position)
    repeat_points_sig = [1, channels] + [1] * (insertion_position - 1) + [channels] + [1] * (
                depth + 1 - insertion_position)
    sig_new_tensor = last_signature_term.unsqueeze(1).repeat(repeat_points_sig)

    A = (new_B * sig_new_tensor).flatten(start_dim=2)
    return A


def solve_optimization_problem(signature, insertion_position, depth, channels):
    """This function solves the optimization problem that allows to approximate the derivatives of the path.

    Arguments:
        signature (:class:`torch.Tensor`): As :func:`signatory.invert_signature`.

        insertion_position (int): Insertion spot.

        depth (int): As :func:`signatory.invert_signature`.

        channels (int): As :func:`signatory.invert_signature`.

    Returns:
        The :class:`torch.Tensor` approximation of the gradient of the streams on the interval indexed by
        :attr:`insertion_position'. It is of shape :math:`(N, \text{channels})'.
    """

    A_matrix = get_insertion_matrix(signature[:, :-channels ** depth], insertion_position, depth - 1, channels)
    b_vector = depth * extract_signature_term(signature, channels, depth)

    x_optimal = torch.matmul(A_matrix, b_vector.unsqueeze(-1)).squeeze(-1)

    sign_1 = extract_signature_term(signature, channels, depth - 1)

    return x_optimal / (torch.norm(sign_1, dim=1).unsqueeze(-1) ** 2)


def invert_signature(signature: torch.Tensor, depth: int, channels: int,
                     initial_position: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Invert the signature with the insertion algorithm: reconstruct a stream of data given its signature. Given that
    the signature is invariant by translation, the initial position of the stream is not recovered.

    The input :attr:`signature` is the signature transform of depth :attr:`depth` of a batch of paths: it should be a
    result from the :func:`signatory.signature` function. The output is a tensor of shape :math:`(N, L, C)`,
    where :math:`N` is the batch size, :math:`L` is the length of the reconstructed stream of data, with
    :math:`L = depth + 1`, and :math:`C` denotes the number of channels.

    Arguments:
        signature (:class:`torch.Tensor`): The signature of a batch of paths, as returned by
            :func:`signatory.signature`. This should be a two-dimensional tensor.

        depth (int): The depth that :attr:`signature` has been calculated to.

        channels (int): The number of channels in the batch of paths that was used to compute :attr:`signature`.

        initial_position (None or :class:`torch.Tensor`, optional): Defaults to None. If it is a :class:`torch.Tensor`
            then it must be of size :math:`(N, C)`, corresponding to the initial position of the paths. If None, the
            reconstructed paths are set to begin at zero.

    Returns:
        The :class:`torch.Tensor` corresponding to a batch of inverted paths.
    """
    if signature_channels(channels, depth) != signature.shape[1]:
        raise ValueError("channels and depth do not correspond to signature shape.")

    batch = signature.shape[0]
    path_derivatives = torch.zeros((batch, depth, channels))
    path = torch.zeros((batch, depth + 1, channels))

    if initial_position is not None:
        path[:, 0, :] = initial_position

    if depth == 1:
        path[:, 1, :] = path[:, 0, :] + signature
    else:
        for insertion_position in torch.arange(1, depth + 1):
            x_optimal = solve_optimization_problem(signature, insertion_position, depth, channels)
            path_derivatives[:, insertion_position - 1, :] = x_optimal
            path[:, insertion_position, :] = (path[:, insertion_position - 1, :]
                                              + path_derivatives[:, insertion_position - 1, :] * (1 / depth))

    return path
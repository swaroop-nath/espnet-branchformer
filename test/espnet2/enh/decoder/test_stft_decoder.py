import pytest
import torch
from packaging.version import parse as V
from torch_complex import ComplexTensor

from espnet2.enh.decoder.stft_decoder import STFTDecoder

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTDecoder_backward(
    n_fft, win_length, hop_length, window, center, normalized, onesided
):
    decoder = STFTDecoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )

    real = torch.rand(2, 300, n_fft // 2 + 1 if onesided else n_fft, requires_grad=True)
    imag = torch.rand(2, 300, n_fft // 2 + 1 if onesided else n_fft, requires_grad=True)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([300 * hop_length, 295 * hop_length], dtype=torch.long)
    y, ilens = decoder(x, x_lens)
    y.sum().backward()


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTDecoder_invalid_type(
    n_fft, win_length, hop_length, window, center, normalized, onesided
):
    decoder = STFTDecoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )
    with pytest.raises(TypeError):
        real = torch.rand(
            2, 300, n_fft // 2 + 1 if onesided else n_fft, requires_grad=True
        )
        x_lens = torch.tensor([300 * hop_length, 295 * hop_length], dtype=torch.long)
        y, ilens = decoder(real, x_lens)


@pytest.mark.skipif(not is_torch_1_12_1_plus, reason="torch.complex32 is used")
@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("win_length", [512])
@pytest.mark.parametrize("hop_length", [128])
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_STFTDecoder_complex32_dtype(
    n_fft, win_length, hop_length, window, center, normalized, onesided
):
    decoder = STFTDecoder(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
    )
    x = torch.rand(
        2,
        300,
        n_fft // 2 + 1 if onesided else n_fft,
        dtype=torch.complex32,
        requires_grad=True,
    )
    x_lens = torch.tensor([300 * hop_length, 295 * hop_length], dtype=torch.long)
    y, ilens = decoder(x, x_lens)
    (y.real.pow(2) + y.imag.pow(2)).sum().backward()

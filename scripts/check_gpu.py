"""Verify NumPy interchange and actual CUDA execution in the selected Python."""

import sys


def main() -> int:
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    try:
        import numpy as np
        import torch

        print("NumPy:", np.__version__)
        print("PyTorch:", torch.__version__)
        print("PyTorch CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the driver and CUDA-enabled PyTorch installation")
        properties = torch.cuda.get_device_properties(0)
        print("GPU:", properties.name)
        print("VRAM GiB:", round(properties.total_memory / 1024**3, 2))
        with torch.no_grad():
            matrix = torch.ones((256, 256), device="cuda:0")
            result = matrix @ matrix
            torch.cuda.synchronize()
            if result[0, 0].item() != 256.0:
                raise RuntimeError("Unexpected CUDA matrix result")
            values = np.array([1, 2, 3], dtype=np.float32)
            output = (torch.from_numpy(values).to("cuda:0") * 2).cpu().numpy()
            np.testing.assert_array_equal(output, np.array([2, 4, 6], dtype=np.float32))
        print("PASS: CUDA matrix computation and NumPy interchange work.")
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

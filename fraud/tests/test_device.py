from types import SimpleNamespace

from fraud import device


def fake(xpu: bool, cuda: bool):
    calls = []
    side = lambda name: SimpleNamespace(is_available=lambda on=(xpu if name == "xpu" else cuda): on, synchronize=lambda: calls.append(f"{name} sync"),
                                        reset_peak_memory_stats=lambda: calls.append(f"{name} reset"), max_memory_allocated=lambda: 2**31,
                                        empty_cache=lambda: calls.append(f"{name} empty"))
    return SimpleNamespace(xpu=side("xpu"), cuda=side("cuda")), calls


def test_xpu_comes_first_so_the_laptop_behaves_as_before(monkeypatch):
    t, _ = fake(True, True)
    monkeypatch.setattr(device, "torch", t)
    assert device.default_device() == "xpu"


def test_cuda_then_cpu(monkeypatch):
    t, _ = fake(False, True)
    monkeypatch.setattr(device, "torch", t)
    assert device.default_device() == "cuda"
    t, _ = fake(False, False)
    monkeypatch.setattr(device, "torch", t)
    assert device.default_device() == "cpu"


def test_calls_follow_the_device(monkeypatch):
    t, calls = fake(True, True)
    monkeypatch.setattr(device, "torch", t)
    device.synchronizer("xpu")()
    device.synchronizer("cuda:0")()
    device.reset_peak_memory("cuda")
    device.reset_peak_memory("xpu")
    assert calls == ["xpu sync", "cuda sync", "cuda reset", "xpu reset"]
    assert device.peak_memory_gib("cuda") == 2.0 and device.peak_memory_gib("xpu") == 2.0


def test_cpu_has_no_peak_and_no_sync(monkeypatch):
    t, calls = fake(False, False)
    monkeypatch.setattr(device, "torch", t)
    device.synchronizer("cpu")()
    device.reset_peak_memory("cpu")
    device.empty_cache()
    assert device.peak_memory_gib("cpu") is None and calls == []

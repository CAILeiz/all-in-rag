"""第三方包的兼容性补丁（不改动 site-packages，只在导入时打 monkey patch）。

问题：当前环境装的是 mistralai 3.x，它是 PEP 420 命名空间包，
既没有顶层 `Mistral`，也没有 `mistralai.models.WandbIntegration` / `ToolCall`。
而 `llama_index.finetuning`（被 `llama_index.experimental` 间接导入）在模块级执行：

    from mistralai import Mistral
    from mistralai.models import (JobsAPIRoutesFineTuning..., WandbIntegration, ...)
    from mistralai.models import ToolCall

于是 `from llama_index.experimental.query_engine import PandasQueryEngine` 直接失败。
本教程并不使用 Mistral，这里注册桩模块让导入链通过；真正调用 Mistral 时会显式报错。
"""
import sys
import types

_STUB_PREFIXES = ("mistralai",)


def _placeholder(fullname: str):
    def _init(self, *args, **kwargs):
        raise RuntimeError(
            f"当前安装的 mistralai 包不含 {fullname}；本教程不会用到 Mistral，"
            "如需使用请安装带完整客户端的 mistralai 版本。"
        )

    return type(fullname.rsplit(".", 1)[-1], (), {"__init__": _init})


def _make_stub(fullname: str) -> types.ModuleType:
    mod = types.ModuleType(fullname)
    mod.__doc__ = "compat_shims 生成的占位模块"

    def _getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        obj = _placeholder(f"{fullname}.{name}")
        setattr(mod, name, obj)
        return obj

    mod.__getattr__ = _getattr

    def _getattr_sub(name, _fullname=fullname):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _make_stub(f"{_fullname}.{name}")
        sys.modules[f"{_fullname}.{name}"] = sub
        setattr(mod, name, sub)
        return sub

    mod.__path__ = []  # 让它看起来像包
    return mod


def patch_mistralai(force: bool = False) -> None:
    """用占位模块替换 mistralai，绕过 llama_index.finetuning 的导入链问题。"""
    installed = sys.modules.get("mistralai")
    if installed is not None and not force:
        if getattr(installed, "__doc__", None) == "compat_shims 生成的占位模块":
            return
        # 真实包：只有当它缺少 Mistral 时才替换
        try:
            import mistralai  # noqa: F401
            from mistralai.models import ToolCall  # noqa: F401
            return  # 真实包可用，什么都不做
        except Exception:
            pass

    for name in list(sys.modules):
        if name == "mistralai" or name.startswith("mistralai."):
            del sys.modules[name]

    root = _make_stub("mistralai")
    root.__path__ = []  # 命名空间包
    sys.modules["mistralai"] = root
    models = _make_stub("mistralai.models")
    sys.modules["mistralai.models"] = models
    root.models = models


def install() -> None:
    patch_mistralai()

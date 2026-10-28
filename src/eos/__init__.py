"""PINN-аппроксимация широкодиапазонного уравнения состояния алюминия."""


def __getattr__(name):
    # EOSModel импортируется лениво: загрузка данных и бейзлайн не тянут torch-модель
    if name == "EOSModel":
        from eos.inference import EOSModel
        return EOSModel
    raise AttributeError(name)

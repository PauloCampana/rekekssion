import typing
import pandas
import numpy
import scipy
import seaborn

class Rekekssion:
    def __init__(
        self,
        intercept,
        y, x,
        n, p,
        Y, X, C, H, B, R, P,
    ):
        self.intercept = intercept
        self.y = y # y name
        self.x = x # x names
        self.n = n # nº rows
        self.p = p # nº columns in X
        self.Y = Y # response
        self.X = X # variables
        self.C = C # cross product XTX
        self.H = H # hat matrix
        self.B = B # coefficients
        self.R = R # residuals
        self.P = P # predictions

    def __repr__(self):
        eq = f"{self.y} = {self.B[0]:.3}"
        if not self.intercept:
            eq += f" {self.x[0]}"
        for i in range(len(self.B)):
            if i == 0:
                continue
            if self.B[i] >= 0:
                eq += f" + {abs(self.B[i]):.3} {self.x[i]}"
            else:
                eq += f" - {abs(self.B[i]):.3} {self.x[i]}"
        return eq

    def _sse(self):
        return numpy.sum(numpy.square(self.R))

    def _mse(self):
        return self._sse() / (self.n - self.p)

    def _ssr(self):
        mean = numpy.mean(self.Y)
        return numpy.sum(numpy.square(self.P - mean))

    def _msr(self):
        return self._ssr() / (self.p - 1)

    def _sst(self):
        mean = numpy.mean(self.Y)
        return numpy.sum(numpy.square(self.Y - mean))

    def _mst(self):
        return self._sst() / (self.n - 1)

    def _mae(self):
        return numpy.mean(numpy.abs(self.R))

    def _rmse(self):
        return numpy.sqrt(self._mse())

    def _rsq(self):
        return 1 - self._sse() / self._sst()

    def _rsq_adj(self):
        return 1 - self._mse() / self._mst()

    def _log_lik(self):
        sigma2 = self._mse()
        l = self.n * numpy.log(2 * numpy.pi * sigma2) + self.n - self.p
        return -0.5 * l

    def _aic(self):
        return 2 * (self.p + 1 - self._log_lik())

    def _bic(self):
        return numpy.log(self.n) * (self.p + 1) - 2 * self._log_lik()

    def _beta_var(self):
        return self._mse() * numpy.diag(self.C)

    def _beta_ic(self, alpha = 0.05):
        t = scipy.stats.t.ppf(1 - alpha / 2, self.n - self.p)
        margin = t * numpy.sqrt(self._beta_var())
        return (self.B - margin, self.B + margin)

    def _tstat(self):
        return self.B / numpy.sqrt(self._beta_var())

    def _tpval(self):
        return scipy.stats.t.sf(numpy.abs(self._tstat()), self.n - self.p)

    def _fstat(self):
        return self._msr() / self._mse()

    def _fpval(self):
        return scipy.stats.f.sf(self._fstat(), self.p, self.n - self.p - 1)

    def summary(self, alpha: float = 0.05):
        print("formula:")
        print(f"\t{self.__repr__()}")
        print("coefficients:")
        biggest_length = 0
        for name in self.x:
            biggest_length = max(biggest_length, len(name))
        print(
            f"\t{'':<{biggest_length}}",
            f"{'estimate':>10}",
            f"{'std err':>10}",
            f"{f'IC {50 * alpha:g}%':>10}",
            f"{f'IC {100 - 50 * alpha:g}%':>10}",
            f"{'statistic':>10}",
            f"{'p-value':>10}",
            sep = "",
        )
        stderr = numpy.sqrt(self._beta_var())
        ic = self._beta_ic(alpha)
        tstat = self._tstat()
        tpval = self._tpval()
        for i in range(len(self.B)):
            line = f"\t{self.x[i]:<{biggest_length}}"
            line += f"{self.B[i]:>10.3f}"
            line += f"{stderr[i]:>10.3f}"
            line += f"{ic[0][i]:>10.3f}"
            line += f"{ic[1][i]:>10.3f}"
            line += f"{tstat[i]:>10.3f}"
            line += f"{tpval[i]:>10.3g}"
            print(line)
        print("metrics:")
        print(f"\tR squared              {self._rsq():>10.3f}")
        print(f"\tadjusted R squared     {self._rsq_adj():>10.3f}")
        print(f"\troot mean squared error{self._rmse():>10.3f}")
        print(f"\tmean absolute error    {self._mae():>10.3f}")
        print(f"\tF test p-value         {self._fpval():>10.3g}")
        print(f"\tAkaike criteria        {self._aic():>10.3f}")
        print(f"\tBayesian criteria      {self._bic():>10.3f}")
        pass

    def coefficients(self):
        return self.B

    def estimated(self):
        return self.P

    def residuals(self, kind: str = "response"):
        match kind:
            case "response":
                return self.R
            case "standardized":
                return self.R / self._rmse()
            case "studentized":
                var = self._mse() * (1 - numpy.diag(self.H))
                return self.R / numpy.sqrt(var)
        raise ValueError(
            "residual kind must be one of: response, standardized, studentized"
        )

    def predict(self, new_data: pandas.DataFrame):
        X = new_data.to_numpy()
        if self.intercept:
            X = numpy.insert(X, 0, 1, axis = 1)
        return X @ self.B

    def plot_qq(self, kind: str, **kwargs):
        r = self.residuals(kind)
        r = sorted(r)
        x = scipy.stats.norm.ppf(
            numpy.linspace(0, 1, self.n),
            loc = numpy.mean(r),
            scale = numpy.sqrt(numpy.var(r))
        )
        return seaborn.relplot(x = x, y = r, linewidth = 0.1, **kwargs) \
            .set_axis_labels("Theoretical quantiles", "Residual quantiles") \
            .ax.axline(
                xy1 = (0, 0), slope = 1,
                color = "#00000040", dashes = (4, 2)
            )

    def plot_normality(self, kind: str, **kwargs):
        r = self.residuals(kind)
        seaborn.displot(x = r, kind = "kde", **kwargs) \
            .set_axis_labels("Residuals")
        x = numpy.linspace(numpy.min(r), numpy.max(r), 1000)
        y = scipy.stats.norm.pdf(
            x,
            loc = numpy.mean(r),
            scale = numpy.sqrt(numpy.var(r))
        )
        seaborn.lineplot(x = x, y = y, color = "#00000040", linestyle = "--")

    def plot_linearity(self, **kwargs):
        return seaborn.relplot(x = self.Y, y = self.P, **kwargs) \
            .set_axis_labels("Observed values", "Estimated values") \
            .ax.axline(
                xy1 = (0, 0), slope = 1,
                color = "#00000040", dashes = (4, 2)
            )

    def plot_homoscedasticity(self, kind: str, **kwargs):
        r = self.residuals(kind)
        return seaborn.relplot(x = self.P, y = r, **kwargs) \
            .set_axis_labels("Estimated values", "Residuals") \
            .ax.axline(
                xy1 = (0, 0), slope = 0,
                color = "#00000040", dashes = (4, 2)
            )

    def plot_autocorrelation(self, kind: str, **kwargs):
        r = self.residuals(kind)
        return seaborn.relplot(x = range(self.n), y = r, **kwargs) \
            .set_axis_labels("Index", "Residuals") \
            .ax.axline(
                xy1 = (0, 0), slope = 0,
                color = "#00000040", dashes = (4, 2)
            )

    def plot_detect_hat(self, **kwargs):
        h = numpy.diag(self.H)
        limit = 2 * self.p / self.n
        return seaborn.relplot(x = self.P, y = h, **kwargs) \
            .set_axis_labels("Estimated values", "Hat values") \
            .ax.axline(
                xy1 = (0, limit), slope = 0,
                color = "#00000040", dashes = (4, 2)
            )

    def plot_detect_cook(self, **kwargs):
        h = numpy.diag(self.H)
        r = self.residuals(kind = "studentized")
        cook = r * r * h / (1 - h) / self.p
        limit = scipy.stats.f.ppf(0.5, self.p, self.n - self.p)
        return seaborn.relplot(x = self.P, y = cook, **kwargs) \
            .set_axis_labels("Estimated values", "Cook's distance") \
            .ax.axline(
                xy1 = (0, limit), slope = 0,
                color = "#00000040", dashes = (4, 2)
            )

    def plot(self, kind: str = "response", **kwargs):
        self.plot_qq(kind, **kwargs)
        self.plot_normality(kind, **kwargs)
        self.plot_linearity(**kwargs)
        self.plot_homoscedasticity(kind, **kwargs)
        self.plot_autocorrelation(kind, **kwargs)
        self.plot_detect_hat(**kwargs)
        self.plot_detect_cook(**kwargs)

def fit(
    data: pandas.DataFrame,
    y: str,
    x: typing.List[str],
    intercept: bool = True,
) -> Rekekssion:
    Y = data[y].to_numpy()
    X = data[x].to_numpy()
    if intercept:
        X = numpy.insert(X, 0, 1, axis = 1)
        x.insert(0, "(intercept)")

    C = numpy.linalg.inv(X.T @ X)
    H = X @ C @ X.T
    B = C @ X.T @ Y
    P = X @ B
    R = Y - P

    n = X.shape[0]
    p = X.shape[1]
    return Rekekssion(intercept, y, x, n, p, Y, X, C, H, B, R, P)

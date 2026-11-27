"""Parte 1: carga, limpieza, folds y evaluación.

Funciones tomadas de notebooks/01_datos_validacion.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# raíz del repo: src/rogii/data.py -> dos niveles arriba de src/rogii
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_well(well, split="train", raw=RAW):
    """Devuelve (horizontal, typewell, ps) de un pozo."""
    base = Path(raw) / split
    hz = pd.read_csv(base / f"{well}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{well}__typewell.csv")
    falta = hz["TVT_input"].isna()
    ps = int(falta.idxmax()) if falta.any() else len(hz)
    return hz, tw, ps


def clean_gr(hz, max_gap=9):
    """Interpola huecos cortos de GR y marca todo lo que faltaba.

    GR_clean: GR con los huecos de hasta max_gap filas interpolados (lineal).
              Los huecos más largos, y los que están en los bordes del pozo, siguen en NaN.
    GR_mask : True donde el GR original faltaba (se haya interpolado o no).
    """
    gr = hz["GR"]
    mask = gr.isna()

    # longitud del hueco al que pertenece cada fila vacía
    grupo = (~mask).cumsum()
    largo_hueco = mask.groupby(grupo).transform("sum")

    # interpola solo entre valores válidos (no extrapola en los bordes)
    interp = gr.interpolate(method="linear", limit_area="inside")

    # deja en NaN lo que cae en huecos largos
    gr_clean = interp.where(~(mask & (largo_hueco > max_gap)))

    out = hz.copy()
    out["GR_clean"] = gr_clean
    out["GR_mask"] = mask
    return out


def smooth_gr(gr_clean, window=9):
    """Mediana móvil centrada. No rellena los huecos que quedaron en NaN."""
    suave = gr_clean.rolling(window, center=True, min_periods=1).median()
    return suave.where(gr_clean.notna())


def resample_typewell(tw, step=0.5, max_gap=9.0):
    """Typewell a grilla regular de TVT (paso `step`, origen en su primer punto).

    GR: interpolación lineal; los huecos de más de max_gap pies quedan en NaN.
    Geology: etiqueta del punto original anterior (solo si el typewell la trae).
    """
    t = tw["TVT"].to_numpy()
    grid = np.arange(t[0], t[-1] + 1e-9, step)

    j = np.clip(np.searchsorted(t, grid + 1e-6, side="right") - 1, 0, len(t) - 1)
    k = np.clip(j, 0, len(t) - 2)
    gap = t[k + 1] - t[k]

    gr = np.interp(grid, t, tw["GR"].to_numpy())
    gr[gap > max_gap + 1e-6] = np.nan
    if "Geology" in tw.columns:
        geo = tw["Geology"].to_numpy()[j]
    else:
        geo = np.full(len(grid), None, dtype=object)
    return pd.DataFrame({"TVT": grid, "GR": gr, "Geology": geo})


def neighbor_groups(vec, n_wells, radio):
    """Etiqueta de grupo de cada pozo, uniendo los pares a distancia <= radio."""
    v = vec[vec["d"] <= radio]
    g = coo_matrix((np.ones(len(v)), (v["a"].to_numpy(int), v["b"].to_numpy(int))),
                   shape=(n_wells, n_wells))
    return connected_components(g, directed=False)[1]


def make_folds(wells_df, grupo, k=5, seed=42):
    """Asigna cada grupo completo a un fold, equilibrando las filas evaluadas.

    grupo: etiqueta de grupo de cada fila de wells_df (mismo orden).
    Devuelve un DataFrame con columnas well, fold.
    """
    peso = wells_df["n_pred"].groupby(grupo).sum()
    # el orden aleatorio solo decide los empates
    orden = (peso.sample(frac=1, random_state=seed)
                 .sort_values(ascending=False, kind="stable"))
    carga = np.zeros(k)
    asignado = {}
    for g, p in orden.items():
        f = int(carga.argmin())
        asignado[g] = f
        carga[f] += p
    return pd.DataFrame({"well": wells_df["well"].to_numpy(),
                         "fold": [asignado[g] for g in grupo]})


def validate_prediction(pred, wells_df, expected_wells=None):
    """Lanza ValueError si `pred` no cumple el formato. Devuelve True si cumple."""
    cols = ["well", "idx", "tvt_pred"]
    if sorted(pred.columns) != sorted(cols):
        raise ValueError(f"columnas {list(pred.columns)}; se esperaban {cols}")
    if len(pred) == 0:
        raise ValueError("la predicción está vacía")
    if pred[cols].isna().any().any():
        raise ValueError("hay valores nulos")
    if not all(isinstance(w, str) for w in pred["well"].unique()):
        raise ValueError("`well` debe ser texto")
    if not pd.api.types.is_integer_dtype(pred["idx"]):
        raise ValueError("`idx` debe ser entero")
    if not pd.api.types.is_numeric_dtype(pred["tvt_pred"]):
        raise ValueError("`tvt_pred` debe ser numérico")
    if not np.isfinite(pred["tvt_pred"].to_numpy(dtype=float)).all():
        raise ValueError("`tvt_pred` tiene valores infinitos")
    if pred.duplicated(["well", "idx"]).any():
        raise ValueError("hay filas (well, idx) duplicadas")

    info = wells_df.set_index("well")
    presentes = set(pred["well"])
    if presentes - set(info.index):
        raise ValueError(f"pozos desconocidos: {sorted(presentes - set(info.index))[:5]}")
    if expected_wells is not None and presentes != set(expected_wells):
        raise ValueError(f"faltan {len(set(expected_wells) - presentes)} pozos "
                         f"y sobran {len(presentes - set(expected_wells))}")

    for w, g in pred.groupby("well")["idx"]:
        ps, n = int(info.at[w, "ps"]), int(info.at[w, "n"])
        # sin duplicados: mínimo, máximo y conteo garantizan el rango completo
        if g.min() != ps or g.max() != n - 1 or len(g) != n - ps:
            raise ValueError(f"{w}: se esperaban las filas {ps} a {n - 1} ({n - ps}); "
                             f"hay {len(g)} entre {g.min()} y {g.max()}")
    return True


def rmse_by_well(pred, truth, wells_df, expected_wells=None):
    """Devuelve (tabla por pozo, RMSE global) sobre el tramo desde el PS.

    truth: columnas well, idx, tvt.
    """
    validate_prediction(pred, wells_df, expected_wells)
    m = pred.merge(truth, on=["well", "idx"], how="left", validate="1:1")
    if m["tvt"].isna().any():
        raise ValueError("hay filas sin TVT real")
    m["se"] = (m["tvt_pred"] - m["tvt"]) ** 2
    por_pozo = m.groupby("well")["se"].agg(n="size", rmse="mean").reset_index()
    por_pozo["rmse"] = np.sqrt(por_pozo["rmse"])
    return por_pozo, float(np.sqrt(m["se"].mean()))


def load_truth(ids_pozos):
    """TVT real desde el PS (solo train)."""
    partes = []
    for w in ids_pozos:
        hz, _, ps = load_well(w)
        partes.append(pd.DataFrame({"well": w, "idx": np.arange(ps, len(hz)),
                                    "tvt": hz["TVT"].to_numpy()[ps:]}))
    return pd.concat(partes, ignore_index=True)


def constant_prediction(tabla):
    """Línea base: TVT constante = último TVT conocido."""
    return pd.concat([pd.DataFrame({"well": r.well, "idx": np.arange(r.ps, r.n),
                                    "tvt_pred": r.tvt_ps})
                      for r in tabla.itertuples()], ignore_index=True)


def to_submission(pred, sample):
    """Convierte al formato de entrega (id, tvt) en el orden de `sample`."""
    s = pd.DataFrame({"id": pred["well"] + "_" + pred["idx"].astype(str),
                      "tvt": pred["tvt_pred"]})
    out = sample[["id"]].merge(s, on="id", how="left", validate="1:1")
    if out["tvt"].isna().any() or len(s) != len(sample):
        raise ValueError("la predicción no coincide con los ids de sample_submission")
    return out

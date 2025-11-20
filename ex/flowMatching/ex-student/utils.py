import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from scipy import interpolate
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve, auc


def plot_1dhistos(data, features, bins=100):
    """
    Plot 1D histograms of the features in the data
    plot both linear and log scale on the right

    features: list of features names 
    if features is N_const, use linspace binning
    """
    n_features = len(features)
    fig, axs = plt.subplots(n_features, 2, figsize=(8, 3 * n_features))

    for i, feature in enumerate(features):
        if feature == 'N_const':
            binning = np.linspace(0, 100, 100)
        else:
            binning = bins
        axs[i, 0].hist(data[:, i], bins=binning, histtype='step', lw=2, alpha=0.7)
        axs[i, 0].set_title(feature)
        axs[i, 1].hist(data[:, i], bins=binning, histtype='step', lw=2, alpha=0.7)
        axs[i, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()
    return

def areas_between_rocs(tpr_real, fpr_real, tpr_flash, fpr_flash, x_lim=0.2):
    if x_lim:
        tpr_real_mask = tpr_real > x_lim
        fpr_real = fpr_real[tpr_real_mask]
        tpr_real = tpr_real[tpr_real_mask]

        tpr_flash_mask = tpr_flash > x_lim
        fpr_flash = fpr_flash[tpr_flash_mask]
        tpr_flash = tpr_flash[tpr_flash_mask]

    # apply log to tpr
    fpr_real_log = np.where(fpr_real < 1e-6, 0, np.log(fpr_real))
    # Step 1: Choose the curve with more data points as the base for x-values (in this case fpr_blue)
    # Step 2: Interpolate the curve with fewer data points to these x-values

    # tpr_flash += np.linspace(0, 1e-6, len(tpr_flash)) # NOT NEEDED IF WE USE NEAREST INTERPOLATION

    interpolator = interpolate.interp1d(
        tpr_flash,
        fpr_flash,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    fpr_flash_interpolated = interpolator(tpr_real)
    fpr_flash_interpolated_log = np.where(
        fpr_flash_interpolated < 1e-6, 0, np.log(fpr_flash_interpolated)
    )

    # Calculate the absolute differences between the curves
    differences = np.abs(fpr_real_log - fpr_flash_interpolated_log)

    # Integrate the differences using Simpson's rule
    area = trapezoid(differences, tpr_real)

    return area


def roc_curve_figure(
    target,
    gen, 
    model=None,
    mode = 'btag',
    y_pred_model=None,
    title="ROC curve",
    perturb=True,
    shade=True,
    ):

    flavour = gen[:, 4]
    mask_light = (flavour == 0)

    if mode == 'btag':
        mask = (flavour == 2)
        target = target[:, 0]

    if mode == 'ctag': 
        mask = (flavour == 1)  
        target = target[:, 5] 
    
    fpr_target, tpr_target, _ = roc_curve(np.concatenate((np.ones_like(target[mask]), np.zeros_like(target[mask_light]))), np.concatenate((target[mask], target[mask_light])))
    roc_auc_target = auc(fpr_target, tpr_target)

    if model is not None:
        if mode == 'btag':
            model = model[:, 0]
        elif mode == 'ctag':
            model = model[:, 5]
        fpr_model, tpr_model, _ = roc_curve(np.concatenate([np.ones_like(model[mask]), np.zeros_like(model[~mask])]), np.concatenate([model[mask], model[~mask]]))
        roc_auc_model = auc(fpr_model, tpr_model)

    # make target and model arrays the same length
    # if model is not None:
    #     max_len = min(len(tpr_target), len(tpr_model))
    #     tpr_target = tpr_target[:max_len]
    #     fpr_target = fpr_target[:max_len]
    #     tpr_model = tpr_model[:max_len]
    #     fpr_model = fpr_model[:max_len]

    if perturb:
        tpr_target_perturbed_minus = 1 - (1 - tpr_target) * 1.2
        tpr_target_perturbed_plus = 1 - (1 - tpr_target) * 0.8

        interpolator = interpolate.interp1d(
            tpr_target,
            fpr_target,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fpr_target_perturbed_minus = interpolator(tpr_target_perturbed_minus)
        fpr_target_perturbed_plus = interpolator(tpr_target_perturbed_plus)

    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.plot(
        tpr_target,
        fpr_target,
        color="dodgerblue",
        lw=1.5,
        label=f"Target",
    )

    if perturb:
        # # shade the area between the perturbed curves
        # # the filling is a grid pattern
        abc_perturbed = areas_between_rocs(
            tpr_target, fpr_target, tpr_target_perturbed_minus, fpr_target, x_lim=0.2
        )

        ax.fill_between(
            tpr_target,
            fpr_target_perturbed_minus,
            fpr_target_perturbed_plus,
            alpha=0.4,
            label="Typ. data vs simulation discrepancy"
            + "\n"
            + f"at LHC, ABC: {abc_perturbed:.3f}",
            color="dodgerblue",
        )

    if model is not None:
        abc = areas_between_rocs(tpr_target, fpr_target, tpr_model, fpr_model, x_lim=0.2)

    if model is not None:
        ax.plot(
            tpr_model,
            fpr_model,
            color="tomato",
            lw=1.5,
            # label=f"model, ABC: {abc:.3f}",
        )
        if shade:
            # cast the two curves to the same length
            interpolator = interpolate.interp1d(
                tpr_model,
                fpr_model,
                kind="nearest",
                bounds_error=False,
                fill_value="extrapolate",
            )
            fpr_model_interpolated = interpolator(tpr_target)
            ax.fill_between(
                tpr_target,
                fpr_target,
                fpr_model_interpolated,
                color="tomato",
                alpha=0.5,
                label=f"model, ABC: {abc:.3f}",
            )

    ax.tick_params(axis="both", which="both", direction="in")
    ax.minorticks_on()
    ax.set_xlabel("True Positive Rate", loc="right", fontsize=14)
    ax.set_ylabel("False Positive Rate", loc="top", fontsize=14)

    ax.set_xlim(0.2, 1)
    ax.set_ylim(1e-4, 1.05)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", alpha=0.5, color="darkgrey", ls="--")
    ax.axvline(x=0.2, color="black", linestyle="--", lw=1.5, alpha=0.6)

    leg = ax.legend(
        frameon=False,
        loc="upper center",
    )

    leg._legend_box.align = "left"
    ax.set_title(f"{title} for {mode}", fontsize=14)

    return fig, ax


def plot_1d_hist(
    flash,
    reco,
    label,
    title,
    rangeHist=None,
    bins=100,
    ratioPlotBounds=(-2, 2),
    logScale=True,
    legendLoc="upper right",
    ymax=None,
):
    # change matplotlib style
    # plt.style.use("fivethirtyeight")
    # if not logScale:
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    ax1.tick_params(which="both", labelbottom=False, direction="in")
    ax1.minorticks_on()
    ax2.tick_params(which="both", direction="in", top=True)
    # ax2.minorticks_on()

    ax2.set_xlabel(label, loc="right", fontsize=14)
    # fig.suptitle(f"{title}", fontsize=20)

    # bins = 100
    # if (
    #     (label == "recoNConstituents")
    #     | (label == "nSV")
    #     | (label == "ncharged")
    #     | (label == "nneutral")
    # ):
    #     bins = np.arange(-0.5, 80.5, 1)

    # Linear scale plot
    _, rangeR, _ = ax1.hist(
        reco, histtype="step", color="dodgerblue", lw=1.5, bins=bins, label="Target"
    )

    saturated_samples = np.where(flash < np.min(rangeR), np.min(rangeR), flash)
    saturated_samples = np.where(
        saturated_samples > np.max(rangeR), np.max(rangeR), saturated_samples
    )
    ax1.hist(
        saturated_samples,
        histtype="step",
        lw=1.5,
        color="tomato",
        range=[np.min(rangeR), np.max(rangeR)],
        bins=bins,
        label=f"Model",
    )
    ax1.set_ylabel("Counts", loc="top", fontsize=14)

    if logScale:
        ax1.set_yscale("log")

    if rangeHist is not None:
        ax1.set_xlim(rangeHist[0], rangeHist[1])

    # Ratio plot for linear scale
    hist_reco, bins_reco = np.histogram(
        reco, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
    )
    hist_flash, bins_flash = np.histogram(
        saturated_samples, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
    )

    wasserstein = stats.wasserstein_distance(reco, flash)
    leg = ax1.legend(
        frameon=False,
        loc=legendLoc,
        title=rf"$WS={wasserstein:.4f}$",
        alignment="left",
        title_fontsize=10,
    )

    # Compute the error on the ratio
    ratio_err = np.sqrt(
        (np.sqrt(hist_flash) / hist_flash) ** 2 + (np.sqrt(hist_reco) / hist_reco) ** 2
    )
    ratio_err = ratio_err * (hist_flash / hist_reco)
    ratio = hist_flash / hist_reco

    # ax2.scatter(bins_reco[:-1], ratio, marker=".", color="black")
    bin_centers = (bins_reco[:-1] + bins_reco[1:]) / 2

    ax2.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_err,
        fmt=".",
        color="black",
        ms=2,
        elinewidth=1,
    )
    ax2.set_ylabel("Model/Target", fontsize=14)
    if rangeHist is not None:
        ax2.set_xlim(rangeHist[0], rangeHist[1])
    ax2.set_ylim(*ratioPlotBounds)
    # horizontal line at 1
    ax2.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    # ax2.text(
    #     s=f"KS: {ks:.3f}",
    #     x=ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.05,
    #     y=ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.8,
    # )
    fig.align_ylabels([ax1, ax2])

    return fig

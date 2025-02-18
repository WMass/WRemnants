import hist
import matplotlib as mpl
import numpy as np
from scipy import stats
from scipy.optimize import nnls

from utilities import parsing
from utilities.styles import styles
from wremnants import histselections as sel
from wremnants import plot_tools
from wremnants import regression as reg
from wremnants.datasets.datagroups import Datagroups
from wums import boostHistHelpers as hh
from wums import logging, output_tools


def plot_chi2(chi2, ndf, suffix=""):
    chi2_flat = chi2.flatten()
    # plot chi2 distributions
    xlim = [0, 50]
    x_chi2 = np.linspace(*xlim, 1000)
    y_chi2 = stats.chi2.pdf(x_chi2, ndf) * len(chi2_flat)

    chi2_total = np.sum(chi2_flat)
    ndf_total = ndf * len(chi2_flat)
    p_total = round(stats.chi2.sf(chi2_total, ndf_total) * 100, 1)

    pvalues = np.array([stats.chi2.sf(c, ndf) for c in chi2_flat])

    logger.info(f"Min(chi2)={chi2_flat.min()}; Max(chi2)={chi2_flat.max()}")
    logger.info(f"Mean(chi2)={chi2_flat.mean()}; std(chi2)={chi2_flat.std()}")
    logger.info(
        f"Total chi2/ndf = {chi2_total}/{ndf_total} = {chi2_total/ndf_total} (p = {p_total}%)"
    )

    chi2label = rf"$\\chi^2/\\mathrm{{ndf}} = {round(chi2_total,0)}/{ndf_total}$"
    plabel = rf"$p={p_total}\%$"

    chi2_flat[chi2_flat > xlim[1]] = xlim[1] - 0.1

    colors = mpl.colormaps["tab10"]

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h,
        ylabel="Entries",
        xlabel=r"$\chi^2$",
        cms_label=args.cmsDecor,
        xlim=xlim,
        ylim=None,
        logy=False,
        rrange=[0.5, 1.5],
        automatic_scale=False,
        rlabel="1/chi2",
    )
    ax2 = ratio_axes[-1]

    fontsize = ax1.xaxis.label.get_size()

    n, bins, _ = ax1.hist(
        chi2_flat,
        bins=50,
        range=xlim,
        color=colors(0),
        label="Entries",
        histtype="step",
    )

    ax1.plot(x_chi2, y_chi2, color="red", label=rf"$\chi^2({ndf})$")

    ax2.plot(xlim, [1, 1], marker="", linestyle="-", color="k")

    bins_center = bins[:-1] + (bins[1:] - bins[:-1]) / 2.0
    chi2_bins = stats.chi2.pdf(bins_center, ndf) * len(chi2_flat)
    ax2.stairs(n / chi2_bins, bins, color=colors(0))

    # ax1.text(0.94, 0.85, binlabel, transform=ax1.transAxes, fontsize=fontsize,
    #         verticalalignment='bottom', horizontalalignment="right")

    ax1.text(
        0.96,
        0.85,
        chi2label,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax1.text(
        0.96,
        0.75,
        plabel,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    ax1.text(
        1.0,
        1.003,
        styles.process_labels[proc],
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plot_tools.addLegend(ax1, ncols=1, text_size=fontsize, loc="upper left")
    plot_tools.fix_axes(ax1, ax2, logy=False)

    outfile = "chi2" + (f"_{suffix}" if suffix != "" else "")
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, args=args)


def plot_pvalues(chi2, ndf, order=1, suffix=""):
    chi2_flat = chi2.flatten()
    # plot chi2 distributions
    xlim = [0, 1]
    x_chi2 = np.linspace(*xlim, 1000)
    y_chi2 = stats.chi2.pdf(x_chi2, ndf) * len(chi2_flat)

    chi2_total = np.sum(chi2_flat)
    ndf_total = ndf * len(chi2_flat)
    p_total = round(stats.chi2.sf(chi2_total, ndf_total) * 100, 1)

    pvalues = np.array([stats.chi2.sf(c, ndf) for c in chi2_flat])

    logger.info(
        f"Min(chi2)={chi2_flat.min()} p={stats.chi2.sf(chi2_flat.min(), ndf)}; Max(chi2)={chi2_flat.max()} p={stats.chi2.sf(chi2_flat.max(), ndf)}"
    )
    logger.info(f"Mean(chi2)={chi2_flat.mean()}; std(chi2)={chi2_flat.std()}")
    logger.info(
        f"Total chi2/ndf = {chi2_total}/{ndf_total} = {chi2_total/ndf_total} (p = {p_total}%)"
    )

    chi2label = f"$\\chi^2/\\mathrm{{ndf}} = {int(round(chi2_total,0))}/{ndf_total}$"
    plabel = rf"Total $p={p_total}\%$"

    colors = mpl.colormaps["tab10"]

    (
        fig,
        ax1,
    ) = plot_tools.figure(
        h,
        ylabel="Frequency",
        xlabel="Probablility",
        cms_label=args.cmsDecor,
        xlim=xlim,
        ylim=[0.0, 2.5],
        logy=False,
        automatic_scale=False,
    )

    fontsize = ax1.xaxis.label.get_size()

    (
        nEntries,
        bins,
    ) = np.histogram(pvalues, bins=20, density=False, range=xlim)

    bincenters = bins[:-1] + (bins[1:] - bins[:-1]) / 2.0
    y = nEntries / sum(nEntries) * 20
    y_err = np.sqrt(nEntries) / sum(nEntries) * 20

    ax1.errorbar(
        bincenters,
        y,
        yerr=y_err,
        color="black",
        linestyle="",
        marker="o",
        label="Entries",
    )

    ax1.plot(xlim, [1, 1], color="red", label=f"Expected")

    ax1.text(
        0.96,
        0.88,
        f"Smoothing order = {order}",
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax1.text(
        0.96,
        0.795,
        plabel,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax1.text(
        0.96,
        0.71,
        chi2label,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    ax1.text(
        1.0,
        1.003,
        styles.process_labels[proc],
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plot_tools.addLegend(ax1, ncols=1, text_size=fontsize, loc="upper left")

    outfile = "pvalues" + (f"_{suffix}" if suffix != "" else "")
    if order != None:
        outfile += f"_smoothingOrder{order}"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, args=args)


def plot_params(params, suffix=""):

    logger.info(f"Min(chi2)={params.min()}; Max(chi2)={params.max()}")
    logger.info(f"Mean(chi2)={params.mean()}; std(chi2)={params.std()}")

    p_mean_label = rf"$\mu = {round(params.mean(), 1)}$"
    p_std_label = rf"$\sigma = {round(params.std(), 1)}$"

    xlim = [params.min(), params.max()]

    colors = mpl.colormaps["tab10"]

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h,
        ylabel="Entries",
        xlabel="Parameter value",
        cms_label=args.cmsDecor,
        xlim=xlim,
        ylim=None,
        logy=False,
        rrange=[0.5, 1.5],
        automatic_scale=False,
        rlabel="1",
    )
    ax2 = ratio_axes[-1]

    fontsize = ax1.xaxis.label.get_size()

    n, bins, _ = ax1.hist(
        params, bins=50, range=xlim, color=colors(0), label="Entries", histtype="step"
    )

    ax2.plot(xlim, [1, 1], marker="", linestyle="-", color="k")

    ax1.text(
        0.96,
        0.85,
        p_mean_label,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax1.text(
        0.96,
        0.75,
        p_std_label,
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    ax1.text(
        1.0,
        1.003,
        styles.process_labels[proc],
        transform=ax1.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plot_tools.addLegend(ax1, ncols=1, text_size=fontsize, loc="upper left")
    plot_tools.fix_axes(ax1, ax2, logy=False)

    outfile = "parameters" + (f"_{suffix}" if suffix != "" else "")
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, args=args)


def plot_diagnostics_extendedABCD(
    h,
    outdir,
    syst_variations=False,
    auxiliary_info=True,
    smoothing_order_spectrum=3,
    smoothing_order_fakerate=2,
    smoothing_mode="full",
):
    # plot the fake distribution in the signal region

    smoothing_axis_name = "pt"
    fakerate_axes = ["eta", "pt", "charge"]

    signal_region = False
    flow = True

    logger.info("Make full fake prediction w/o rebinning")
    selector = sel.FakeSelector1DExtendedABCD(
        h,
        fakerate_axes=fakerate_axes,
        smoothing_order_spectrum=smoothing_order_spectrum,
        smoothing_order_fakerate=smoothing_order_fakerate,
        smoothing_mode=smoothing_mode,
        # throw_toys="normal",
    )

    h = hh.rebinHist(h, selector.name_y, h.axes[selector.name_y].edges[:2])
    h = hh.rebinHist(h, selector.name_x, h.axes[selector.name_x].edges[:3])

    if smoothing_mode == "full":
        # get values and variances of all sideband regions (this assumes signal region is at high abcd-x and low abcd-y axis bins)
        sval = h.values(flow=flow)
        svar = h.variances(flow=flow)
        # move abcd axes last
        idx_x = h.axes.name.index(selector.name_x)
        idx_y = h.axes.name.index(selector.name_y)

        sval = np.moveaxis(sval, [idx_x, idx_y], [-2, -1])
        svar = np.moveaxis(svar, [idx_x, idx_y], [-2, -1])

        # invert y-axis to get signal region last
        sval = np.flip(sval, axis=-1)
        svar = np.flip(svar, axis=-1)

        if signal_region:
            sval = sval.reshape((*sval.shape[:-2], sval.shape[-2] * sval.shape[-1]))[
                ..., -1
            ]
            svar = svar.reshape((*svar.shape[:-2], svar.shape[-2] * svar.shape[-1]))[
                ..., -1
            ]
        else:
            # make abcd axes flat, take all but last bin (i.e. signal region D)
            sval = sval.reshape((*sval.shape[:-2], sval.shape[-2] * sval.shape[-1]))[
                ..., :-1
            ]
            svar = svar.reshape((*svar.shape[:-2], svar.shape[-2] * svar.shape[-1]))[
                ..., :-1
            ]

        smoothidx = [
            n for n in h.axes.name if n not in [selector.name_x, selector.name_y]
        ].index(selector.smoothing_axis_name)
        smoothing_axis = h.axes[selector.smoothing_axis_name]
        nax = sval.ndim

        # underflow and overflow are left unchanged along the smoothing axis
        # so we need to exclude them if they have been otherwise included
        if flow:
            smoothstart = 1 if smoothing_axis.traits.underflow else 0
            smoothstop = -1 if smoothing_axis.traits.overflow else None
            smoothslice = slice(smoothstart, smoothstop)
        else:
            smoothslice = slice(None)

        slices = nax * [slice(None)]
        slices[smoothidx] = smoothslice

        sval = sval[*slices]
        svar = svar[*slices]

        xwidth = h.axes[selector.smoothing_axis_name].widths

        xwidthtgt = xwidth[
            *smoothidx * [None], :, *(nax - smoothidx - 2 + signal_region) * [None]
        ]
        xwidth = xwidth[*smoothidx * [None], :, *(nax - smoothidx - 1) * [None]]

        sval *= 1.0 / xwidth
        svar *= 1.0 / xwidth**2

        goodbin = (sval > 0.0) & (svar > 0.0)
        goodbin = goodbin & ~(
            (sval < 1.0) & (svar / sval**2 <= 1.0)
        )  # exclude bins with 0 data entries but negative prompt MC subtraction
        if goodbin.size - np.sum(goodbin) > 0:
            logger.warning(
                f"Found {goodbin.size-np.sum(goodbin)} of {goodbin.size} bins with 0 or negative bin content, those will be set to 0 and a large error"
            )

        y = np.where(goodbin, np.log(sval), 0.0)
        y_var = np.where(goodbin, svar / sval**2, np.inf)
        x = selector.get_bin_centers_smoothing(
            h, flow=True
        )  # the bins where the smoothing is performed (can be different to the bins in h)

        # transform with weights
        w = 1 / np.sqrt(y_var)

        # move smoothing axis to last
        axes = [
            n
            for n in h.axes.name
            if n
            not in [
                selector.name_x,
                selector.name_y,
                *selector.fakerate_integration_axes,
            ]
        ]
        idx_ax_smoothing = axes.index(selector.smoothing_axis_name)
        if idx_ax_smoothing != len(axes) - 1:
            y = np.moveaxis(y, idx_ax_smoothing, -1)
            w = np.moveaxis(w, idx_ax_smoothing, -1)

        regressor = selector.spectrum_regressor
        # smooth
        regressor.solve(x, y, w)

        # compute chi2 from individual sideband regions
        y_pred = regressor.evaluate(x)
        y_pred = y_pred.reshape(y.shape)  # flatten
        w = w.reshape(y.shape)
        chi2, ndf = reg.compute_chi2(y, y_pred, w, nparams=regressor.params.shape[-1])

        plot_pvalues(chi2, ndf, order=smoothing_order_spectrum)

        for idx_region, region in enumerate(["Ax", "Bx", "A", "B", "C"]):
            plot_pvalues(
                chi2[..., idx_region],
                ndf,
                order=smoothing_order_spectrum,
                suffix=f"region{region}",
            )

        chi2_cmax = chi2[..., -2].flatten()

        # return

        # # FIXME
        # regressor.params = regressor.params[...,0,:]
        # regressor.cov = regressor.cov[...,0,:,:]

        # regions = ['Ax', ]#'Bx', 'A', 'B', 'C', 'D']

        # y_pred_d = regressor.evaluate(x)
        # y_pred_d_var = regressor.get_eigenvector_predictions(x)

        # # return

        # add up parameters from smoothing of individual sideband regions
        # exp(ax + 2*b - bx -2*a + c)
        # ['ax', 'bx', 'a', 'b', 'c']
        w_region = np.array([1, -1, -2, 2, 1], dtype=int)
        # linear parameter combination
        regressor.params = np.sum(
            regressor.params
            * w_region[
                *[np.newaxis] * (regressor.params.ndim - 2), slice(None), np.newaxis
            ],
            axis=-2,
        )

        regressor.cov = np.sum(
            regressor.cov
            * w_region[
                *[np.newaxis] * (regressor.params.ndim - 2),
                slice(None),
                np.newaxis,
                np.newaxis,
            ]
            ** 2,
            axis=-3,
        )

        if regressor.polynomial == "monotonic":

            # performing a nnls to enforce monotonicity for the signal region (using generalized least squares)
            Y = regressor.params
            W = np.linalg.inv(regressor.cov.reshape(-1, *regressor.cov.shape[-2:]))
            W = W.reshape((*regressor.cov.shape[:-2], *W.shape[-2:]))
            WY = np.einsum("...ij,...j->...i", W, Y)
            # the design matrix X is just a 1xn unity matrix and can thus be ignored
            XTWY = WY
            XTWX = W

            orig_shape = XTWY.shape
            nBins = np.prod(orig_shape[:-1])
            XTWY_flat = XTWY.reshape(nBins, XTWY.shape[-1])
            XTWX_flat = XTWX.reshape(nBins, XTWX.shape[-2], XTWX.shape[-1])
            regressor.params = [
                nnls(xtwx, xtwy)[0] for xtwx, xtwy in zip(XTWX_flat, XTWY_flat)
            ]
            regressor.params = np.reshape(regressor.params, orig_shape)

            # allow the integration constaint to be negative
            if np.sum(regressor.params[..., 0] == 0) > 0:
                mask = regressor.params[..., 0] == 0
                mask_flat = mask.flatten()
                w_flip = np.ones(XTWY.shape[-1])
                w_flip[0] = -1
                params_negative = [
                    nnls(xtx, xty)[0]
                    for xtx, xty in zip(
                        XTWX_flat[mask_flat], XTWY_flat[mask_flat] * w_flip
                    )
                ]
                regressor.params[mask] = np.array(params_negative) * w_flip
                logger.info(
                    f"Found {mask.sum()} parameters that are excluded in nnls and negative"
                )

        # for ip in range(params_d.shape[-1]):
        #     p = params[...,ip].flatten()
        #     p_d = params_d[...,ip].flatten()
        #     plot_params(p, suffix=f"order{smoothing_order_spectrum}_power{ip}")
        #     plot_params(p_d, suffix=f"regionD_order{smoothing_order_spectrum}_power{ip}")

        regions = ["Ax", "Bx", "A", "B", "C", "D"]
        # regions = ['C',]

        y_pred_d = regressor.evaluate(x)
        y_pred_d_var = regressor.get_eigenvector_predictions(x)

    elif smoothing_mode == "fakerate":
        y, y_var = selector.compute_fakeratefactor(
            h, smoothing=False, syst_variations=False
        )
        y_pred, y_pred_var = selector.compute_fakeratefactor(
            h, smoothing=True, syst_variations=True
        )

        selection = {n: hist.sum for n in selector.fakerate_integration_axes}
        hRebin = (
            hh.rebinHist(
                h[selection],
                selector.smoothing_axis_name,
                selector.rebin_smoothing_axis,
            )
            if selector.rebin_smoothing_axis is not None
            else h[selection]
        )

        xRebin_edges = hRebin.axes[smoothing_axis_name].edges
        xRebin_widths = np.diff(xRebin_edges) / 2
        xRebin_centers = xRebin_edges[:-1] + xRebin_widths

        regressor = selector.fakerate_regressor

        yRebin_pred = regressor.evaluate(xRebin_centers)
        yRebin_pred_var = regressor.get_eigenvector_predictions(xRebin_centers)

        w = 1 / np.sqrt(y_var)

        # smoothing axis must be last
        chi2, ndf = reg.compute_chi2(
            np.moveaxis(y, 1, -1),
            yRebin_pred,
            np.moveaxis(w, 1, -1),
            nparams=regressor.params.shape[-1],
        )

        plot_pvalues(chi2, ndf, order=smoothing_order_fakerate, suffix=f"regionFR")

        regions = ["FR"]

    etavar = r"\eta"
    logy = False
    linestyles = ["-", "-", "--", "--", ":"]
    colors = mpl.colormaps["tab10"]

    x_edges = h.axes[smoothing_axis_name].edges
    x_widths = np.diff(x_edges) / 2
    x_centers = x_edges[:-1] + x_widths

    for idx_charge, charge_bins in enumerate(h.axes["charge"]):
        logger.info(f"Now at charge bin {idx_charge}")
        for idx_eta, eta_bins in enumerate(h.axes["eta"]):
            logger.info(f"Now at eta bin {idx_eta}")
            for idx_region, region in enumerate(regions):
                if region != "C":
                    continue
                slices = {"eta": idx_eta, "charge": idx_charge}

                outfile = f"charge{idx_charge}_eta{idx_eta}_region{region}"
                binlabel = f"$\\mathrm{{{region}}}: {round(eta_bins[0],1)} < {etavar} < {round(eta_bins[1],1)}$"

                if region in "D":
                    yy_pred = y_pred_d[idx_eta, idx_charge]
                    yy_pred_var = y_pred_d_var[idx_eta, idx_charge]
                    chi2label = ""
                elif region == "FR":
                    yy = y[idx_eta, :, idx_charge]
                    yy_err = y_var[idx_eta, :, idx_charge] ** 0.5
                    yy_pred = y_pred[idx_eta, :, idx_charge]
                    yy_pred_var = y_pred_var[idx_eta, :, idx_charge]

                    yyRebin_pred = yRebin_pred[idx_eta, idx_charge, :]
                    yyRebin_pred_var = yRebin_pred_var[idx_eta, idx_charge, :]

                    chi2_ = chi2[idx_eta, idx_charge]
                    chi2label = f"$\\chi^2/\\mathrm{{ndf}} = {round(chi2_,1)}/{ndf}$"
                else:
                    yy = y[idx_eta, idx_charge, idx_region]
                    yy_err = y_var[idx_eta, :, idx_charge, idx_region] ** 0.5
                    yy_pred = y_pred[idx_eta, idx_charge, idx_region]
                    chi2_ = chi2[idx_eta, idx_charge, idx_region]
                    chi2label = f"$\\chi^2/\\mathrm{{ndf}} = {round(chi2_,1)}/{ndf}$"

                    # yy_pred = y_pred_d[idx_eta, idx_charge]
                    # yy_pred_var = y_pred_d_var[idx_eta, idx_charge]

                xlim = [26, 56]

                if region == "D":
                    ylim = [np.min(yy_pred_var), np.max(yy_pred_var)]
                elif region == "FR":
                    ylim = [
                        min(min(yy - yy_err), np.min(yy_pred_var)),
                        max(max(yy + yy_err), np.max(yy_pred_var)),
                    ]
                else:
                    ylim = [min(yy), max(yy)]
                yrange = ylim[1] - ylim[0]
                ylim = [ylim[0] - yrange * 0.1, ylim[1] + yrange * 0.25]

                yy_err[yy_err == np.inf] = yrange * 2

                fig, ax1, ratio_axes = plot_tools.figureWithRatio(
                    h,
                    ylabel="log(Events)" if region != "FR" else "Events",
                    xlabel=styles.xlabels.get(smoothing_axis_name, smoothing_axis_name),
                    cms_label=args.cmsDecor,
                    xlim=xlim,
                    ylim=ylim,
                    logy=logy,
                    rrange=args.rrange if region != "D" else [0.94, 1.06],
                    rlabel=(
                        "pulls"
                        if region not in ["D", "FR"]
                        else "var/nominal" if region == "D" else "1/binned"
                    ),
                    # rlabel="1/nominal",
                )
                ax2 = ratio_axes[-1]

                fontsize = ax1.xaxis.label.get_size()

                if region == "D":
                    ax1.errorbar(
                        x_centers,
                        yy_pred,
                        xerr=x_widths,
                        marker="",
                        linestyle="none",
                        color="k",
                        label="f(x)",
                    )
                    for i in range(yy_pred_var.shape[-2]):
                        ax1.stairs(
                            yy_pred_var[:, i, 0],
                            x_edges,
                            linestyle="-",
                            color=colors(i),
                        )  # down
                        ax1.stairs(
                            yy_pred_var[:, i, 1],
                            x_edges,
                            linestyle="--",
                            color=colors(i),
                        )  # up
                elif region == "FR":
                    ax1.errorbar(
                        xRebin_centers,
                        yy,
                        yerr=yy_err,
                        marker=".",
                        linestyle="none",
                        color="k",
                        label="binned",
                    )

                    ax1.errorbar(
                        x_centers,
                        yy_pred,
                        xerr=x_widths,
                        marker="",
                        linestyle="none",
                        color="k",
                        label="f(x)",
                    )
                    for i in range(yy_pred_var.shape[-2]):
                        ax1.stairs(
                            yy_pred_var[:, i, 0],
                            x_edges,
                            linestyle="-",
                            color=colors(i),
                        )  # down
                        ax1.stairs(
                            yy_pred_var[:, i, 1],
                            x_edges,
                            linestyle="--",
                            color=colors(i),
                        )  # up
                else:
                    ax1.errorbar(
                        x_centers,
                        yy,
                        yerr=yy_err,
                        marker=".",
                        linestyle="none",
                        color="k",
                        label="binned",
                    )
                    ax1.errorbar(
                        x_centers,
                        yy_pred,
                        xerr=x_widths,
                        marker="",
                        linestyle="none",
                        color="red",
                        label="f(x)",
                    )

                    # #FIXME
                    # ax1.errorbar(x_centers, yy_pred, xerr=x_widths, marker="", linestyle='none', color="k", label="f(x)")
                    # for i in range(yy_pred_var.shape[-2]):
                    #     ax1.stairs(yy_pred_var[:,i,0], x_edges, linestyle='-', color=colors(i)) # down
                    #     ax1.stairs(yy_pred_var[:,i,1], x_edges, linestyle='--', color=colors(i)) # up

                # ratios

                # ax2.errorbar(x, yy/yy_pred, xerr=x_widths, yerr=yy_err/yy_pred, marker="", linestyle='none', color=colors(0))
                if region == "D":
                    ax2.plot(xlim, [1, 1], marker="", linestyle="-", color="k")
                    for i in range(yy_pred_var.shape[-2]):
                        ax2.stairs(
                            yy_pred_var[:, i, 0] / yy_pred,
                            x_edges,
                            linestyle="-",
                            color=colors(i),
                        )  # down
                        ax2.stairs(
                            yy_pred_var[:, i, 1] / yy_pred,
                            x_edges,
                            linestyle="--",
                            color=colors(i),
                        )  # up
                elif region == "FR":
                    ax2.plot(xlim, [1, 1], marker="", linestyle="-", color="k")
                    ax2.stairs(
                        yyRebin_pred / yyRebin_pred,
                        xRebin_edges,
                        linestyle="-",
                        color="k",
                    )  # down
                    for i in range(yyRebin_pred_var.shape[-2]):
                        ax2.stairs(
                            yyRebin_pred_var[:, i, 0] / yyRebin_pred,
                            xRebin_edges,
                            linestyle="-",
                            color=colors(i),
                        )  # down
                        ax2.stairs(
                            yyRebin_pred_var[:, i, 1] / yyRebin_pred,
                            xRebin_edges,
                            linestyle="--",
                            color=colors(i),
                        )  # up
                    # data
                    ax2.errorbar(
                        xRebin_centers,
                        yy / yyRebin_pred,
                        yerr=yy_err / yyRebin_pred,
                        marker=".",
                        linestyle="none",
                        color="k",
                    )
                else:
                    # #FIXME
                    # ax2.plot(xlim, [1,1], marker="", linestyle='-', color="k")
                    # ax2.stairs(yy_pred/yy_pred, x_edges, linestyle='-', color="k") # down
                    # for i in range(yy_pred_var.shape[-2]):
                    #     ax2.stairs(yy_pred_var[:,i,0]/yy_pred, x_edges, linestyle='-', color=colors(i)) # down
                    #     ax2.stairs(yy_pred_var[:,i,1]/yy_pred, x_edges, linestyle='--', color=colors(i)) # up
                    # # data
                    # ax2.errorbar(x_centers, yy/yy_pred, yerr=yy_err/yy_pred, marker=".", linestyle='none', color="k")

                    ax2.plot(xlim, [0, 0], marker="", linestyle="-", color="k")
                    ax2.errorbar(
                        x_centers,
                        (yy - yy_pred) / yy_err,
                        yerr=np.ones_like(x_centers),
                        marker=".",
                        linestyle="none",
                        color="k",
                    )

                ax1.text(
                    0.94,
                    0.85,
                    binlabel,
                    transform=ax1.transAxes,
                    fontsize=fontsize,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )

                ax1.text(
                    0.94,
                    0.75,
                    chi2label,
                    transform=ax1.transAxes,
                    fontsize=fontsize,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )

                ax1.text(
                    1.0,
                    1.003,
                    styles.process_labels[proc],
                    transform=ax1.transAxes,
                    fontsize=fontsize,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                )
                plot_tools.addLegend(ax1, ncols=1, text_size=fontsize, loc="upper left")
                plot_tools.fix_axes(ax1, ax2, logy=logy)

                if args.postfix:
                    outfile += f"_{args.postfix}"

                plot_tools.save_pdf_and_png(outdir, outfile)
                plot_tools.write_index_and_log(outdir, outfile, args=args)

            # return


if __name__ == "__main__":
    parser = parsing.plot_parser()
    parser.add_argument(
        "infile",
        help="Output file of the analysis stage, containing ND boost histograms",
    )
    parser.add_argument(
        "-n",
        "--baseName",
        type=str,
        help="Histogram name in the file (e.g., 'nominal')",
        default="nominal",
    )
    parser.add_argument(
        "--procFilters",
        type=str,
        nargs="*",
        default=[
            "Fake",
        ],
        help="Filter to plot (default no filter, only specify if you want a subset",
    )
    parser.add_argument(
        "--rrange",
        type=float,
        nargs=2,
        default=[-2.5, 2.5],
        help="y range for ratio plot",
    )

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    groups = Datagroups(args.infile, excludeGroups=None)

    logger.info(f"Load fakes")
    groups.loadHistsForDatagroups(
        args.baseName, syst="", procsToRead=args.procFilters, applySelection=False
    )

    histInfo = groups.getDatagroups()

    for proc in args.procFilters:
        h = histInfo[proc].hists[args.baseName]

        # plot_diagnostics_extendedABCD(h, outdir, smoothing_order_spectrum=2)
        plot_diagnostics_extendedABCD(
            h, outdir, smoothing_order_spectrum=3
        )  # , smoothing_order_fakerate=3, smoothing_mode="fakerate")
        # plot_diagnostics_extendedABCD(h, outdir, smoothing_order_spectrum=4)

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)

# Call-site style of the shared `comparisonplot` in the NetworkOutbreaks vignettes (owner: WP34, V-NO).
#
# The recipe's default size, (380n + 60) × 620 px for n observable columns, keeps Plots' default font sizes, so
# a three-column figure scaled to the text width has tick labels of about 4 pt, and its bottom-left y-label
# ("model − SSA mean") is clipped by the left edge. `cmp_style(n)` gives a narrower canvas, larger fonts and a
# left margin for that label; `cmp_style(n; title = true)` also leaves room for a `plot_title` above the panel
# titles. Use it as `comparisonplot(ref, det; observables, cmp_style(length(observables))...)`.
using Plots
const _MM = Plots.PlotMeasures.mm      # (Catalyst also exports `mm`)

# Font sizes of the ordinary single-panel figures (Plots' defaults, 8 pt ticks on a 600 × 400 canvas, are small once
# the figure is scaled to the text width); `cmp_style` sets its own.
default(tickfontsize = 10, guidefontsize = 12, legendfontsize = 9, titlefontsize = 13)

function cmp_style(n::Integer; title::Bool = false, legend = :topright)   # legend: one value or [top bottom]
    width = n == 1 ? 640 : n == 2 ? 820 : 400n
    big = n >= 3                       # three columns are scaled down most: larger fonts
    base = (size = (width, (big ? 740 : 660) + (title ? 40 : 0)), left_margin = (big ? 11 : 9) * _MM,
            bottom_margin = 2 * _MM, right_margin = 2 * _MM, tickfontsize = big ? 13 : 11,
            guidefontsize = big ? 14 : 12, titlefontsize = big ? 15 : 13, legendfontsize = n == 1 ? 9 : 8,
            legend = legend)
    return title ? merge(base, (plot_titlefontsize = big ? 16 : 14, plot_titlevspan = n == 1 ? 0.06 : 0.08,
                                top_margin = 1 * _MM)) : base
end

# The legends of the first column are about as wide as a panel and sit at the top right; `legend_room!` raises the
# upper y-limit of the top-left panel by `factor` of its range, and that of the bottom-left (residual) panel by
# `bottom` of its range, so that the legends do not cover the curves. Use it as `comparisonplot(...) |> legend_room!`.
function legend_room!(p; factor = 1.5, bottom = 1.3)
    n = length(p.subplots) ÷ 2
    for (k, f) in ((1, factor), (n + 1, bottom))
        lo, hi = Plots.ylims(p[k])
        Plots.ylims!(p[k], (lo, lo + f * (hi - lo)))
    end
    return p
end

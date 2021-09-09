# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31

@author: kruu

Plot script for go-arounds generation marginal densities
"""

import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()

# Data 
df_real = pd.read_pickle("Data/distributions_along_lines.pkl")
df_generated = pd.read_pickle("Data/generated_GM_and_sampling.pkl")

def marginals(df_real, df_generated):

    data_real = df_real.melt(ignore_index=False).rename(columns=dict(variable="step")).reset_index(level=0, inplace=False)
    data_gen = df_generated.melt(ignore_index=False).rename(columns=dict(variable="step")).reset_index(level=0, inplace=False)

    # Brush for selection
    selec1 = alt.selection_multi(encodings=["color"], init=[{"index": data_real.index.values[0]}])
    color1 = alt.condition(selec1, alt.Color("index:N", legend=None), alt.value("lightgrey"))
    opacity1 = alt.condition(selec1, alt.value(1), alt.value(0.1))

    selec2 = alt.selection_multi(encodings=["color"], init=[{"index": data_gen.index.values[0]}])
    color2 = alt.condition(selec2, alt.Color("index:N", legend=None), alt.value("lightgrey"))
    opacity2 = alt.condition(selec2, alt.value(1), alt.value(0.1))

    chart1 = alt.Chart(data_real, title = "Real Data").mark_point().encode(
        alt.Row("step"), alt.X("value"), color = color1, opacity = opacity1
    ).add_selection(selec1)

    chart2 = alt.Chart(data_gen, title = "Generated Data").mark_point().encode(
        alt.Row("step"), alt.X("value"), color = color2, opacity = opacity2
    ).add_selection(selec2)

    return chart1 | chart2

def ridgeline(df):

    data = df.melt().rename(columns=dict(variable="step"))

    step = 20
    overlap = 1

    chart = alt.Chart(data, height=step).mark_area(
        interpolate="monotone", fillOpacity=0.8, stroke="lightgray", strokeWidth=0.5
    ).encode(
        alt.Row(
            "step:N",
            title=None,
            header=alt.Header(
                labelAngle=0, labelAlign="left", labelFont="Ubuntu", labelFontSize=14
            ),
        ),
        alt.X("bin_min:Q", axis=None),
        alt.Y("count:Q", axis=None, scale=alt.Scale(range=[step, -step * overlap])),
        alt.Fill(
            "std:Q",
            legend=None,
            scale=alt.Scale(domain=(5000, 0), scheme="redyellowblue"),
        ),
    ).transform_joinaggregate(
        std="stdev(value)",
        groupby=["step"],
    ).transform_bin(
        ["bin_max", "bin_min"], "value", bin=alt.Bin(maxbins=30)
    ).transform_joinaggregate(
        count="count()",
        groupby=["step", "bin_min", "bin_max"],
    ).transform_impute(
        impute="count", groupby=["step", "std"], key="bin_min", value=0
    ).properties(
        title="Ridgeline plot of marginal distributions", bounds="flush"
    ).configure_facet(
        spacing=0,
    ).configure_view(
        stroke=None
    ).configure_title(
        anchor="start", font="Ubuntu", fontSize=16
    ).configure_axis()

    return chart

marginals(df_real, df_generated).save('marginals.html', embed_options={'renderer':'svg'})
ridgeline(df_real).save('ridgeline_true.html', embed_options={'renderer':'svg'})
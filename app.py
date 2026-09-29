if run_comparison:
    results = []
    for t in multi_tickers:
        data = get_stock_data(t, period)
        if data is not None and not data.empty:
            prev_date, prev_close, next_date, next_day, change, r2 = predict_stock(data)
            results.append([t, prev_date, prev_close, next_date, next_day, change, r2])

    if results:
        comp_df = pd.DataFrame(results, columns=["Ticker","Prev Date","Prev Close","Next Date","Predicted Close","Change%","R²"])

        # ✅ Color-coded Predicted Close column
        def highlight_pred(val, prev):
            return f"color: {'green' if val > prev else 'red'}; font-weight:bold;"
        styled_df = comp_df.style.apply(
            lambda row: [highlight_pred(row["Predicted Close"], row["Prev Close"]) 
                         if col=="Predicted Close" else "" for col in comp_df.columns],
            axis=1
        )

        # Centered bold heading
        st.markdown("<div class='section-title'>📊 Multi‑Ticker Comparison</div>", unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True)

        # ✅ Interactive bar chart for predicted vs previous close
        fig1 = px.bar(comp_df.melt(id_vars="Ticker", value_vars=["Prev Close","Predicted Close"],
                                   var_name="Type", value_name="Price"),
                      x="Ticker", y="Price", color="Type", barmode="group",
                      title="Predicted vs Previous Close",
                      labels={"Price":"Price","Ticker":"Stock"},
                      hover_data={"Price":True,"Type":True})
        st.plotly_chart(fig1, use_container_width=True)

        # ✅ Interactive bar chart for percentage change
        fig2 = px.bar(comp_df, x="Ticker", y="Change%", color="Change%",
                      title="Predicted Percentage Change",
                      labels={"Change%":"% Change","Ticker":"Stock"},
                      color_continuous_scale=["red","green"],
                      hover_data={"Change%":True,"R²":True})
        st.plotly_chart(fig2, use_container_width=True)

        # ✅ Legend/explanation
        st.markdown("<p style='text-align:center; font-size:16px;'>Green = Profit, Red = Loss. Hover over bars for details.</p>", unsafe_allow_html=True)

        # ✅ Download comparison
        st.download_button("📥 Download Comparison (CSV)", comp_df.to_csv(index=False).encode("utf-8"), "comparison.csv", "text/csv")

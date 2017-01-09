import CarbonCycleModel

if __name__ == "__main__":
    model = CarbonCycleModel.CarbonCycleModel()

    model.print_sink_analysis()
    model.print_forcing3_analysis()
    model.print_forcing2_analysis()

    model.plot_forcing3()
    model.plot_forcing2()

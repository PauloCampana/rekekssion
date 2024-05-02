import os
import pandas

dirname = os.path.dirname(__file__)

class data:
    biomass = pandas.read_csv(os.path.join(dirname, "biomass.csv"));
    concrete = pandas.read_csv(os.path.join(dirname, "concrete.csv"));
    cpu = pandas.read_csv(os.path.join(dirname, "cpu.csv"));

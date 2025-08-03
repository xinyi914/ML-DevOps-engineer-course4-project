import logging
import pandas as pd
# Add the necessary imports for the starter code.


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info("Load data")
df = pd.read_csv("../data/census.csv")
logger.info("Clean data")
df.columns = df.columns.str.strip()
df_clean = df.map(lambda x: x.replace(" ","") if isinstance(x,str) else x)

# save clean data
logger.info("Save clean data")
df_clean.to_csv("../data/census_clean.csv")

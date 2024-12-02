# TDS-project

the project uses python 3.11.1

reference to dataset: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data

The downloaded dataset is inside data folder.

data insights:
recency - Number of days since customer's last purchase. however if customer didn't purchase anything it means enrollment date.
should drop in marital status: Absurd , YOLO  values.

NumDealsPurchases = NumWebPurchases + NumCatalogPurchases + NumStorePurchases
income - yearly. currency - can't be found.
goldProducts = premium products, not a real gold. (like premium meat)
zColumns - irrelavant and should be dropped
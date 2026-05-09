import pandas as pd
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
products=["Laptop","Phone","Tablet","HeadPhones","Smartwatch"]
categories={
    "Laptop":"Electronics",
    "Phone":"Electronics",
    "Tablet":"Electronics",
    "HeadPhones":"Accessories",
    "Smartwatch":"Accessories"
}

columns=["Time","Product","Category","Price","Quantity"]
sales_data=pd.DataFrame(columns=columns)

def generate_sales():
    product=random.choice(products)
    price=random.randint(200,1500)
    quantity=random.randint(1,5)
    data={"Time":datetime.now().strftime("%H:%M:%S"),
          "Product":product,"Category":categories[product],
          "Price":price,"Quantity":quantity}
    return data

for i in range(1,21):
    new_sale=generate_sales()
    sales_data.loc[len(sales_data)]=new_sale
    print("New Transaction:",new_sale)
    time.sleep(2)

sales_data["Revenue"]=sales_data["Price"]*sales_data["Quantity"]
total_revenue=sales_data["Revenue"].sum()
print("Total Sale:",total_revenue)

most_sold=sales_data["Product"].value_counts()
print("Most Sold Product")
print(most_sold)

category_sales=sales_data.groupby("Category")["Quantity"].sum()
print(category_sales)

product_sales=sales_data.groupby("Product")["Quantity"].sum()
product_sales.plot(kind="bar")
plt.title("Product Sales Analysis")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.show()

sales_data.to_csv("live_sales_data.csv",index=False)
print("Sales data saved to live_sales_data.csv")
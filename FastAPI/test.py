# import sqlite3

# connection = sqlite3.connect("house_price.db")
# conn = connection.cursor()



# # conn.execute(f"INSERT INTO house_price" + \
# # f" VALUES (155,1,2,3,4,5,6,7,8);")

# # connection.commit()
# # conn.close()
# # connection.close()

# #conn.execute("DELETE  from house_price;")

# conn.execute("select *  from house_price;")

# results = conn.fetchall()

# for r in results:
#     print(r)
#
# connection.commit()
# conn.close()
# connection.close()


import pandas as pd
df= pd.read_pickle('test_data.pkl')
df = df.reset_index(drop=True)
#df= read_from_pickle('test_data.pkl')
#print(df)
one = df.loc[0]
print(one['MedInc'])
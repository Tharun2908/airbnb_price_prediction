#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[3]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# In[7]:


df = pd.read_csv("/app/data/clean_airbnb_data.csv")




# In[8]:


X = df.drop('price',axis=1)
y = df["price"]


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


scaler = StandardScaler()


# In[11]:


X_train_scaled = scaler.fit_transform(X_train)


# In[12]:


X_test_scaled = scaler.transform(X_test)


# In[16]:


X_train_scaled.shape


# In[17]:


X_test_scaled.shape


# In[21]:


model = Sequential()

model.add(Dense(31,activation='relu'))
model.add(Dense(31,activation='relu'))
model.add(Dense(31,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse',metrics=['mae'])


# In[22]:


from tensorflow.keras.callbacks import EarlyStopping


# In[23]:


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,             
    restore_best_weights=True
)


# In[24]:


model.fit(x=X_train_scaled,y=y_train.values,
          validation_split=0.2,
          validation_data=(X_test_scaled,y_test.values),
          batch_size=64,
          callbacks=[early_stop],
          epochs=200,
          verbose=1)


# In[25]:


losses = pd.DataFrame(model.history.history)


# In[27]:


losses[['loss','val_loss']].plot()


# In[38]:


y_pred_nn = model.predict(X_test_scaled).flatten()


# In[39]:


mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)


# In[40]:


print("📈 Neural Network Results:")
print(f"MAE:  {mae_nn:.2f}")
print(f"RMSE: {rmse_nn:.2f}")
print(f"R²:   {r2_nn:.2f}")





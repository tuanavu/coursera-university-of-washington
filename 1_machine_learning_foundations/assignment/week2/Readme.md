## Predicting house prices

In this module, we focused on using regression to predict a continuous value (house prices) from features of the house (square feet of living space, number of bedrooms,...). We also built an iPython notebook for predicting house prices, using data from King County, USA, the region where the city of Seattle is located.

In this assignment, we are going to build a more accurate regression model for predicting house prices by including more features of the house. In the process, we will also become more familiar with how the Python language can be used for data exploration, data transformations and machine learning. These techniques will be key to building intelligent applications.

Follow the rest of the instructions on this page to complete your program. When you are done, **_instead of uploading your code, you will answer a series of quiz questions_** (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

#### **Learning outcomes**

- Execute programs with the iPython notebook
- Load and transform real, tabular data
- Compute summaries and statistics of the data
- Build a regression model using features of the data

#### **Resources you will need**

- Make sure you have downloaded and installed Python, iPython notebook and GraphLab Create. [You can find the instructions here](https://eventing.coursera.org/api/redirectStrict/3DVlLmaXvUKhzIORLv3sT0CsJeyqiWEMRhxuoepq6rDWWjfJy423P_UHp6Mx7vT-toa6SDUSXyhTllNLGapO1A.nKQg32wdrDDA3uYfxp3_WQ.h6s8eEsdn4fmLizq1qcyJ7OA-8h8FIS-YuKbdpub40u8OKW78cVtiN3aM56iUqCr2V0gqbR3czn7CPabm1vM0zRw_OTWDES6M8b1BtaZgHAr2dbC8Gv2qSuJj6Dc32eCCIlZOfD70aLrhAWDQ2e6asIfLdLk1NeRkNYPpnq_rrl6WJJ3E3gglfVJpGlVFWlY9gQXO4sjMg1rr2AdN4fSUt6iQOfhPDYC5Sa7CatSsp6Kz2VKR1Krz8R9V7CK4HlJBrgkrLMkSTJ3nj3DkG5HKT-R2xn4vtWjrkTH9pBiOf8). _(If you are using an ML package other than GraphLab Create, please see the note below.)_
- There are many Python resources available online. [Here is a good place for documentation](https://eventing.coursera.org/api/redirectStrict/9Xs9GSJYRYu0vkT_BX83B6bThYJHRa4YLjmh_DhoADdf8OPWW7mMzb_UsnWQAcboWUtsgetaz-4UXMc63MlGqg.hmWY64tp9xB4e8Z9hdNlOA.Pj5SC8PCZ_XnMtvGFhNl-DkDwqa9lvY0ge07zvtHWAUCzVutqcQhvkT1GWaOaIYLAJwBRmx9rGm6WvRkR6ExtjnV4RY6EI-GSpnmdUZ1q3tVMOOizS1GWzNcqL7_GO6ucuFhbNblVtXZrPOTRl-j7OP-uiSi3Jmtailnafu5T9aqel2BcoImLEQYdEe4eCmJhC0F73L7JhOU5Y2ENNvyV-bWqgvSZe_8I-6YZimw4ykRv6eI_5niIn6Rxa7YZgIhNUbfx_7hMsYGJ250J6akl5gLdh_hSlEVwhBS8objhDtxThGj66N2S_381ogQsntG).
- For GraphLab Create, there is also a lot of information available online. Here are some starting points.

 Learning Concepts about the Tools [https://dato.com/learn/](https://eventing.coursera.org/api/redirectStrict/NGNqKpv6GqgLCM4qDsIOF_nFO9TENkXBYFJkFFZo3k2FaKMEHWeATkslF12pJ6l8keoDNaWiCt3HUyZ6WdVNWQ.zRVuH_Vdn4y4SXthVA5rDw.oZ7gjJAowoVOKN-n20qJQ2SzaommpJC42L8j0N7v3YC94VT-B8BkK4MDCZtcC-liCXJ0UwLwqGkINu6MvEbBV2kiBzCrjev2FhknLNmMX15YY2PatLaF2WSlV7f4VsTGofQ5qDdX-MMi15FuqXc7iTM3kIEHx6TQ3ukMz1zY4D4U7kIdJUiMyZAxMwELO-p9mH03GXfNxwa-PgxXk2kuEy00NMtn3MB8ZEPAzLvqz1JZ4viJ8IyIM-_iky_r3YT1luslXH8vN3Gpm0jazr-NAw)

 The User Guide [https://dato.com/learn/userguide/](https://eventing.coursera.org/api/redirectStrict/GOTzgo-bTrpyhvNfPjcsERd949oMZi6nKstVDWNvRxJ3Fmitl3fxYOSwBvJuzBy_UudXaS6kajzZXZHCgO7aVQ.YBXbE037Md-ff-xgFJUZbw.0KoaM6ncoZNIEbalgS-0DkDY1ZjtHyXfEzag5_fAESCBmoxWjJMqUmJ8XIsEzsSulwDIvUg9jSm_7GUwfjjfgiJ2tYlmqUxrMU4C-E-yWDrwEgdzsHR1YkOFElskl_Xo5QcOrxpo56oxsX9E0sAVltdrsr8xSQI-xuQwWdcwkHisWlZg7ILG1yuM1cqbG7BjSY2NN_vft1T2b_aXnQXpAm-bIgar4SQxNmR4HbfOztt_EhSJr9GuwissqzB91K3Q0gALf_qgEeCNbgEJ5GpFLi4cWpZQDB4ZZRJhm8f1D9g)

 More Detailed API Docs [https://dato.com/products/create/docs/](https://eventing.coursera.org/api/redirectStrict/P_VxYH6nqSKNPQM02vw8qOtvpFOfdGq2peeyu4ovhIXjOMp3O8yOXFfHdegQIVI0nF1GxOAwJEvzA9IjSYXbdA.emzkJYLqzmsCzIV3Jhfcdg.vvugKT0hIazMiQ35_k5NaN15U98hctWMjkW6WI4eKxBG4F7DuqT4WLgruBGyESM9shcolk3MXXzqzn-VO7gc1MbryNVl3d0m4z8eJE7Ze7fHvyLDjV1t4v9_2WtiwayGA2XIyzK-a89ZZGUJFTwoigO1mxHvD48WLNHwikrHH_3lQ3q9-MoO_u7VLhHRNe6TIQeSZhgdQgG8U5lC_61ZbpqCd6GZlkGu7-RbBms4h5hdfZtQiWwsypHmCk9ZOMhTWnp1j_XJi_mdz0FvaiFax1Ck5OFArg4fIF_ErA9ao6qdzFCFTF_wsA1OPzoqzCeT)

#### Download the data and starter code to use GraphLab Create

Before getting started, you will need to download the dataset and the starter iPython notebook that we used in the module.

- Download the house sales pricing dataset here, in SFrame format:** [home_data.gl.zip](https://eventing.coursera.org/api/redirectStrict/i-oNoBk3eJVxXqc5umlg3sj8ILeMtSgGynwj6BAAHdd2Y1u75D8Uo3awcTPp7EZSads1EbAkLqmXQ_q_YTnj7Q.OrU3ydn96kcBiwBFuCOl5Q.hESEH6Z4ZNDfHIWfaz-l6y2wG6oUbjAF4wGJdOglOGvQ8Ih_cKi5_nvQecTmO-tlHwmR5Zg6xElHpYfMFern1ddF0G5jkFwnm8eB4bwZlDwHdoDo6fkGXfIYJ03ATsNnXofIHvD0KuGKwJPjHRmMH0yNK4CpmWudwUjsg-hQXzgMIzUf6EAZPRRnbNd_Tf767GAfSA1daGoBdbayb2loeGQ-xvlbPoZJdlVlyc_-wBSuNpxflq77IM7IxcbYT1yGIiw7cYv7DdtSXEhbjekt-f_ZwLhuf4C0O1TQ5E9AZC29Vnr8kW2etmkRDKdo01OVCwcsJjgE3V8e88-nWjzbvPBMmGU4pIAS7Dbe5chJuAn-Psg9MtCT1KQI-L_78NyKr8F4VAIyjadJZeI0bAWLcKxbJ6ZH8bJymyOk8yVBj5zlkcY4r-S0VWIR2lAwTQCY)**
- Download the house price prediction notebook from the module here: **[Predicting house prices.ipynb](https://eventing.coursera.org/api/redirectStrict/IQsM_nnzeqPfLNcdAUtLcO47gR9jdTn6tYyaRbu3BjWglDSEecC7zrrckkvKaaewn6VPiVUA0odxu6n5yHFJ8A.KyMq7ojQtH60r70qtcpjCQ.dPLAXXILbp-BXjOn9NQ93UFsev9mLM7XhgZaUfVvqnr6oBqk2yKJ3Je1LKkyZd5okio-PBmk74MLrVrT1Wfiwb9Amko0Oy-2JPt4T5YQm1e_5PFSmXG47eCQQ3Eqlt-pznDrT_J7Wzwd2lILYDIsn6_z7uA7Lw7ygl9gXd0eOyqQrJ387loeKRGberOXkhZrE7p6Qvb5up-wYpTpH6-pWvdFrCmSwXlUGvqCLfC0TlRxA6mQFX1aX8ANdHmRjIyO6FmgXMhEDvrFjn64IyFJ-uSjM3vClS8lROS0Vha4N_0nH-RECggXJvXQwk6uM3-s-aebfzl_g-Glgm0a33m3fbfFzfvYp9Omr6uM2tMa92XL2CR-SK-OVu8z263b_sDb5a8GHD2feQFtB-cuLwRDXdMY-BaSw0_LHxhRIg_eJhc3rISff96k7mOkAwXkKODTJ7ntW14WjBLVlP6sxkZzqAfuu7h1zX1M3d28mXiDDlg)**
- Save both of these files in the same directory (where you are calling iPython notebook from) and unzip the data file.

Now you are ready to get started!

#### _**Note: If you would rather use other ML tools...**_

You are welcome to use any ML tool for this course, such as [scikit-learn](https://eventing.coursera.org/api/redirectStrict/HCCSnH1nQHmAehu_Vy2rUTx4fJJbUI6f-YJGW-UDaa9rCAFnt4EyjXrKOgTvq_WRAiAVRblZ00wInXqOSCvwDQ.0zedBIbTLtmORKxp9A8C6A.sNBxk0KznNPvOgIpRocwCS6SHZqg8dllWsYH5UKRbSCtnrCNhgBcZZqx7CqCvvIwVsWToj7PlOH5-fSxJhMgPpYgDbNmAKr5KsxcE9D9QzFTPSBIJMwOj0VlVSDTxKUKdz9hg76-T_4lxX7jnd3_u-u1kV7Su6igkQKIVuBm0ie6ZlKhL17SgOmjluHMRTuubSaLUjtdDneTq0lObs-vKDFG8pv068nnnRXDikmAtMqpyfyZoK38GrsnhFFCpjhGDILgWko_VvByZw5W4-RGFIuec6xmitk8wqXdVwBFAWQ). Though, as discussed in the intro module,_ we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)_

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in [open-source under a permissive BSD license](https://eventing.coursera.org/api/redirectStrict/L7Prp5nrLfZeKs5rnViO8buQtSfWGev90yspozY7zE4hBtQdIEsNN_y4fxhPL0Q2AwiKRElumbkdIHTFVWNpBQ.O_bxzTMGI0abI0hz35Dn7w.k8k4NHDUyyfUMJcuOyLUtJqi_ULDwHlXwAY0nPS6e-qoIbtaYbHDOlI7KbOTIINrqRrR4TcKqEdzkAYbSkczi4bMo7_G-Vm6HHPEmYmYzXD_LoR6melkpS-Urzw6-TJ8IFvu8h5F3RJoE8h3Fj-qTtcGRH-uv9g6HeYMP9ssERQ8gVTt-hMcj3OvSGLzu5wGe-Y2nYi5S1QxRW8A0mixyuooBvsxRRrVENyx5kvGQ7E53lc2NsHZjbUk62ASSJhlpQrLdmfi0UFVzhhqw1lF2zSuDI3FNLu-x5XtEZ52lYw). So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use [Pandas](https://eventing.coursera.org/api/redirectStrict/Kr9Rk1Nj03JDKgOEffRlvDXpq5yyZDr1TssWlU-w4ktbSfkeMqiDGvlbmzi3W-lhnEY3kbapWXxGmhwq2shWcg.hTDY7AF5S1fZVSL3LcVnAA.JN2MJl_VZQVvUJBaGWkcRrJH0vlhOqcchtMMGXhUxiPXktA1fvBAJT7bxgiD_YbUUOM9p1n0lrOvj_sGkVF_CLDoDykn--f7i0FRZ3XLYcb3PQWF4fLXyPLN1EDqV82bbyt0d43l8Iz8nXFzmab645VbhNeK1GLREPV1URMS3AztNloSKfNkzSqsnZRbVol_Ai2A6yVtOFqliac3y05Mnr8b2humyUfbaMgu6oB9MqkzlV0zRNx8siBIgoyCTK-GU_pwp4wrd_gpU4mVXXJFnQ) or other options out there: [home_data.csv](https://eventing.coursera.org/api/redirectStrict/sqwV4UEWZCkrp5JX8PlQ7atNNScPimlmKNoo_LHkNHbs8kkeBNFQtvALBOzqH8dKqUSZW0VWDSRw2vud9gLouw.qcK4tqlDSYkCW1AifCv47A.TZg4q9bMgSUnnCuoH4G1cuXLRvmt_I7iCMDokeN-qHF-biBJQOhQSV6rSY85vaCo6edKbTEMuv1RsOALjDTsMMGrQ2FPiQQ0sl-xq_0ckK-cTsAnug2AHTo5wC7b38IULGQhfVdIUs0IMGb05agnI1t70Zr-dO7Uy7AUVJyKXG7YdoJRAr0DyE2R98FxKJEG9djWJ5hZinpOdd39ndujBSEeBGKRGHQ9qdXD-D7LpTpDzNd6kmFeglHV3DatsQO9482O7C32kWvOAiPklZfIzxYx2Qz_7fvFO8puWZeYZVNaUbd_PGGawtzTqDgP98CzXaFFKg7q83vywJBRwbkZ4paFHKTG91TybHofds5yCyAZ2kv6x3neYVnUoqJWnoEW)

#### Watch the video and explore the iPython notebook on predicting house prices

If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebook on predicting house prices from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in this example.

#### What you will do

Now you are ready! We are going do three tasks in this assignment. There are 3 results you need to gather along the way to enter into the quiz after this reading.

**1. Selection and summary statistics:** In the notebook we covered in the module, we discovered which neighborhood (zip code) of Seattle had the highest average house sale price. Now, take the sales data, select only the houses with this zip code, and compute the average price. **_Save this result to answer the quiz at the end._**

**_2. _****Filtering data:** One of the key features we used in our model was the number of square feet of living space (‘sqft_living’) in the house. For this part, we are going to use the idea of filtering (selecting) data.

- In particular, we are going to use logical filters to select rows of an SFrame. You can find more info in the[Logical Filter section of this documentation](https://eventing.coursera.org/api/redirectStrict/u_ATn753qAYAB9uVWyuLnPtJjht1SWpN2n_ZNHJNJ6Wzwl1WNn1m6delRFnwqJq0-V3Kay49qqaIKIkwFnprUw.5VKKUnXqwTxJjZUj9DeFvw.OPlgT9170CnKukrFzszEvCFgQag-u5R3uNplZ-GVPnZNFnfVqCblBXgHwnJTOTNI6gMARNGCFL2eDCBaojBzBGicjfUg-5Ck6hm38RyA5TiwSaofe-JPQtG0m7strIZExZrrKtWwS4PpY7U9M20Zbdwf2t4L_Jf5FKHoHKe2ABWvqsykgCatuap8RPF73KeGx3z7oztz7KNviWlx6NHrFJadHm2lBZvX0jafwrSHxX0fn5yXXES9vAytczmJvhcZJDBIp2COWwWBNcPjaAbFBDrADsM1R9mq_7G2HOsgst16ccRtb2frXi3Xq6fSt1z0Zy5_AHlDo1HUzdHVW88hneyYNPySGBL3QS6gW3dBMl91kr466Dm-M5eOCN3TFNs2aBLayzLDxkfEpRYYvML3Ww).
- Using such filters, first select the houses that have ‘sqft_living’ higher than 2000 sqft but no larger than 4000 sqft.
- What fraction of the all houses have ‘sqft_living’ in this range? **_Save this result to answer the quiz at the end._**

**3. Building a regression model with several more features:** In the sample notebook, we built two regression models to predict house prices, one using just ‘sqft_living’ and the other one using a few more features, we called this set

```
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
```

Now, you will build a model using the following features:

```
advanced_features = 
[
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]
```

**_Note that using copy and paste from this webpage to the IPython Notebook sometimes does not work perfectly in some operating systems, especially on Windows. For example, the quotes defining strings may not paste correctly. Please check carefully is you use copy & paste._**

- **Compute the RMSE** (root mean squared error) on the test_data for the model using just _my_features_, and for the one using _advanced_features_.

Note 1: when doing the train-test split, make sure you use seed=0, so you get the same training and test sets, and thus results, as we do.

Note 2: in the module we discussed residual sum of squares (RSS) as an error metric for regression, but GraphLab Create uses root mean squared error (RMSE). These are two common measures of error regression, and RMSE is simply the square root of the RSS:

RMSE can be more intuitive, since its units are the same as that of the target column in the data, in our case the unit is dollars ($).

- **What is the difference in RMSE between the model trained with my_features and the one trained with advanced_features?** **_Save this result to answer the quiz at the end._**
## Recommending songs

## 

In this module, we focused on building recommender systems to find products, music and movies that interest users. We also built an exciting iPython notebook for recommending songs, which compared the simple popularity-based recommendation with a personalized model, and showed the significant improvement provided by personalization.

In this assignment, we are going to explore the song data and the recommendations made by our model. In the process, you are going to learn how to use one of the most important data manipulation primitives, _groupby_. These techniques will be important to building the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, **_instead of uploading your code, you will answer a series of quiz questions_** (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

#### Learning outcomes

## 

- Execute song recommendation code with the iPython notebook
- Load and transform real, song data
- Build a song recommender model
- USe the model to recommend songs to individual users
- Use groupby to compute aggregate statistics of the data

#### Resources you will need

## 

- Make sure you have downloaded and installed Python, iPython notebook and GraphLab Create. [You can find the instructions here](https://eventing.coursera.org/api/redirectStrict/3DVlLmaXvUKhzIORLv3sT0CsJeyqiWEMRhxuoepq6rDWWjfJy423P_UHp6Mx7vT-toa6SDUSXyhTllNLGapO1A.nKQg32wdrDDA3uYfxp3_WQ.h6s8eEsdn4fmLizq1qcyJ7OA-8h8FIS-YuKbdpub40u8OKW78cVtiN3aM56iUqCr2V0gqbR3czn7CPabm1vM0zRw_OTWDES6M8b1BtaZgHAr2dbC8Gv2qSuJj6Dc32eCCIlZOfD70aLrhAWDQ2e6asIfLdLk1NeRkNYPpnq_rrl6WJJ3E3gglfVJpGlVFWlY9gQXO4sjMg1rr2AdN4fSUt6iQOfhPDYC5Sa7CatSsp6Kz2VKR1Krz8R9V7CK4HlJBrgkrLMkSTJ3nj3DkG5HKT-R2xn4vtWjrkTH9pBiOf8). _(If you are using an ML package other than GraphLab Create, please see the note below.)_
- There are many Python resources available online. [Here is a good place for documentation](https://eventing.coursera.org/api/redirectStrict/9Xs9GSJYRYu0vkT_BX83B6bThYJHRa4YLjmh_DhoADdf8OPWW7mMzb_UsnWQAcboWUtsgetaz-4UXMc63MlGqg.hmWY64tp9xB4e8Z9hdNlOA.Pj5SC8PCZ_XnMtvGFhNl-DkDwqa9lvY0ge07zvtHWAUCzVutqcQhvkT1GWaOaIYLAJwBRmx9rGm6WvRkR6ExtjnV4RY6EI-GSpnmdUZ1q3tVMOOizS1GWzNcqL7_GO6ucuFhbNblVtXZrPOTRl-j7OP-uiSi3Jmtailnafu5T9aqel2BcoImLEQYdEe4eCmJhC0F73L7JhOU5Y2ENNvyV-bWqgvSZe_8I-6YZimw4ykRv6eI_5niIn6Rxa7YZgIhNUbfx_7hMsYGJ250J6akl5gLdh_hSlEVwhBS8objhDtxThGj66N2S_381ogQsntG).
- For GraphLab Create, there is also a lot of information available online. Here are some starting points.

 Learning Concepts about the Tools [https://dato.com/learn/](https://eventing.coursera.org/api/redirectStrict/NGNqKpv6GqgLCM4qDsIOF_nFO9TENkXBYFJkFFZo3k2FaKMEHWeATkslF12pJ6l8keoDNaWiCt3HUyZ6WdVNWQ.zRVuH_Vdn4y4SXthVA5rDw.oZ7gjJAowoVOKN-n20qJQ2SzaommpJC42L8j0N7v3YC94VT-B8BkK4MDCZtcC-liCXJ0UwLwqGkINu6MvEbBV2kiBzCrjev2FhknLNmMX15YY2PatLaF2WSlV7f4VsTGofQ5qDdX-MMi15FuqXc7iTM3kIEHx6TQ3ukMz1zY4D4U7kIdJUiMyZAxMwELO-p9mH03GXfNxwa-PgxXk2kuEy00NMtn3MB8ZEPAzLvqz1JZ4viJ8IyIM-_iky_r3YT1luslXH8vN3Gpm0jazr-NAw)

 The User Guide [https://dato.com/learn/userguide/](https://eventing.coursera.org/api/redirectStrict/GOTzgo-bTrpyhvNfPjcsERd949oMZi6nKstVDWNvRxJ3Fmitl3fxYOSwBvJuzBy_UudXaS6kajzZXZHCgO7aVQ.YBXbE037Md-ff-xgFJUZbw.0KoaM6ncoZNIEbalgS-0DkDY1ZjtHyXfEzag5_fAESCBmoxWjJMqUmJ8XIsEzsSulwDIvUg9jSm_7GUwfjjfgiJ2tYlmqUxrMU4C-E-yWDrwEgdzsHR1YkOFElskl_Xo5QcOrxpo56oxsX9E0sAVltdrsr8xSQI-xuQwWdcwkHisWlZg7ILG1yuM1cqbG7BjSY2NN_vft1T2b_aXnQXpAm-bIgar4SQxNmR4HbfOztt_EhSJr9GuwissqzB91K3Q0gALf_qgEeCNbgEJ5GpFLi4cWpZQDB4ZZRJhm8f1D9g)

 More Detailed API Docs [https://dato.com/products/create/docs/](https://eventing.coursera.org/api/redirectStrict/P_VxYH6nqSKNPQM02vw8qOtvpFOfdGq2peeyu4ovhIXjOMp3O8yOXFfHdegQIVI0nF1GxOAwJEvzA9IjSYXbdA.emzkJYLqzmsCzIV3Jhfcdg.vvugKT0hIazMiQ35_k5NaN15U98hctWMjkW6WI4eKxBG4F7DuqT4WLgruBGyESM9shcolk3MXXzqzn-VO7gc1MbryNVl3d0m4z8eJE7Ze7fHvyLDjV1t4v9_2WtiwayGA2XIyzK-a89ZZGUJFTwoigO1mxHvD48WLNHwikrHH_3lQ3q9-MoO_u7VLhHRNe6TIQeSZhgdQgG8U5lC_61ZbpqCd6GZlkGu7-RbBms4h5hdfZtQiWwsypHmCk9ZOMhTWnp1j_XJi_mdz0FvaiFax1Ck5OFArg4fIF_ErA9ao6qdzFCFTF_wsA1OPzoqzCeT)

#### Download the data and starter code

## 

Before getting started, you will need to download the dataset and the starter iPython notebook that we used in the module.

- Download the wikipedia dataset with articles on famous people here in SFrame format: **[song_data.gl.zip](https://eventing.coursera.org/api/redirectStrict/vuABM96k-8D2uliNCrxxxU-cYZBjLym8KJC8zRPFTObMumc_LlT1N8_oGTzret2mFuG3ESKJ7AT3xqoAwWYFFg.aE-4OHL9h4ofMsysPtYWlg.TJTn_rtVvmCSbSQLC8KFPA7tubiYczJ57vUhAPgzmt69iPPIMIH8xDmXRgglEercW70NhUAU_9TcHXSEIb84qkoc6kYSgQ6Wc4ZRAB1g-NkgBMuEdxYWU3ccEefeHD-BF8m312tVBRC-S7NaXxpMRdGJHnqfM7zMp3J_dvjk0Y5nb4HK2tfaWWOLuJXmpJglVoOqCnXTGTFk8FePamlUO_Y6T3evMkMZB1w8T2d2mTyKHK187zQaUofaxXMFIflt2OJTlevE4tWNhTQY5fItzawlpNZ6EWm8VVxhPbBTRkqK8p4wjDJp8Ix3MuF3h93Bemo1X1hawzhLr-kewsrEB5RVIyAouiVtE3XBIrcq_whiNRJZuilqQe0dAw2vkHIgaHlt3XUeAVLDRE2ptZg_iY0xMsNAg3RZF7VHIPLVJ7D24xvVCtofBlOUHSfFSDKd)**
- Download the document retrieval notebook from the module here: [Song recommender.ipynb](https://eventing.coursera.org/api/redirectStrict/kmF_xCYrWPOoVhC__Ezm5DQyjzVU6UpdFWmM9wrQnDL-eqwkbVJVfr2sVk9MxvEyKk9buxsodErtTIZ3s8RKVg.SJdrWz3KaPFf3KQrNiwv3A.3VlnoYP_duHC3xDDSpVf8sH_Y0oV-1X6YVO0r4nohik38svXxVS2-a8lm1pIL4erAPfOvQyZvFKJ-d230icOab02V9-CnW4PANC-qtw1UZzmivBijI3rDSz_Kn3RXecSPuI4fcDmH1hhzz9yFQ2zxk_Vp8DsNAvElxjPBmqzNSQUJ0J-MKWX-_y_ucHPnHsow9iJNjFSha0T_fQp19B55hOimmnHeJdPBbjr4ADxWnIGXjBiGQdFHFP2dnoytaRkcRcmnYTZlcLe_kNpYV9SZhvI9EVgqm5z5acTnnrpENFTr5kAWUZI-xYEEbBaq2h_aNOJh3rn3HdnWP6U7bcQuJ-KXB9vGE3vWZCHxPswkElX0WRWioIj9VYbnJbELn3Ko4b3xKLu6nv8JZhxqMHeNuVzXAtzRHHi1FYcP3hROWyf6mEPhopkvunb4RZ7nSaJACuT_tkF-x9IuuDF9U0eZw)
- Save both of these files in the same directory (where you are calling iPython notebook from) and unzip the data file.

Now you are ready to get started!

#### _**Note: If you would rather use other ML tools...**_

## 

You are welcome to use any ML tool for this course, such as [scikit-learn](https://eventing.coursera.org/api/redirectStrict/HCCSnH1nQHmAehu_Vy2rUTx4fJJbUI6f-YJGW-UDaa9rCAFnt4EyjXrKOgTvq_WRAiAVRblZ00wInXqOSCvwDQ.0zedBIbTLtmORKxp9A8C6A.sNBxk0KznNPvOgIpRocwCS6SHZqg8dllWsYH5UKRbSCtnrCNhgBcZZqx7CqCvvIwVsWToj7PlOH5-fSxJhMgPpYgDbNmAKr5KsxcE9D9QzFTPSBIJMwOj0VlVSDTxKUKdz9hg76-T_4lxX7jnd3_u-u1kV7Su6igkQKIVuBm0ie6ZlKhL17SgOmjluHMRTuubSaLUjtdDneTq0lObs-vKDFG8pv068nnnRXDikmAtMqpyfyZoK38GrsnhFFCpjhGDILgWko_VvByZw5W4-RGFIuec6xmitk8wqXdVwBFAWQ). Though, as discussed in the intro module,_ we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)_

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in [open-source under a permissive BSD license](https://eventing.coursera.org/api/redirectStrict/L7Prp5nrLfZeKs5rnViO8buQtSfWGev90yspozY7zE4hBtQdIEsNN_y4fxhPL0Q2AwiKRElumbkdIHTFVWNpBQ.O_bxzTMGI0abI0hz35Dn7w.k8k4NHDUyyfUMJcuOyLUtJqi_ULDwHlXwAY0nPS6e-qoIbtaYbHDOlI7KbOTIINrqRrR4TcKqEdzkAYbSkczi4bMo7_G-Vm6HHPEmYmYzXD_LoR6melkpS-Urzw6-TJ8IFvu8h5F3RJoE8h3Fj-qTtcGRH-uv9g6HeYMP9ssERQ8gVTt-hMcj3OvSGLzu5wGe-Y2nYi5S1QxRW8A0mixyuooBvsxRRrVENyx5kvGQ7E53lc2NsHZjbUk62ASSJhlpQrLdmfi0UFVzhhqw1lF2zSuDI3FNLu-x5XtEZ52lYw). So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use [Pandas](https://eventing.coursera.org/api/redirectStrict/Kr9Rk1Nj03JDKgOEffRlvDXpq5yyZDr1TssWlU-w4ktbSfkeMqiDGvlbmzi3W-lhnEY3kbapWXxGmhwq2shWcg.hTDY7AF5S1fZVSL3LcVnAA.JN2MJl_VZQVvUJBaGWkcRrJH0vlhOqcchtMMGXhUxiPXktA1fvBAJT7bxgiD_YbUUOM9p1n0lrOvj_sGkVF_CLDoDykn--f7i0FRZ3XLYcb3PQWF4fLXyPLN1EDqV82bbyt0d43l8Iz8nXFzmab645VbhNeK1GLREPV1URMS3AztNloSKfNkzSqsnZRbVol_Ai2A6yVtOFqliac3y05Mnr8b2humyUfbaMgu6oB9MqkzlV0zRNx8siBIgoyCTK-GU_pwp4wrd_gpU4mVXXJFnQ) or other options out there: [song_data.csv](https://eventing.coursera.org/api/redirectStrict/tYB7mgIFvr1mpIiJUJ_72C_avahWf8nays19MCd43lPrrLhQIOJZ1rfXZIXept1a8BrrVg2EwjRhgcPt5IjK3g.ikTNF_wOV4oABBvjQ8Zi_Q.Q8RIfhYq-x70JUBeZ5aKzO-o3LD53OCVopdDk9jrcuYWzEQT9aDxGTHlVUrP-nkruChN2HwVEBlBA9gdOe77uXRNbG7qyQbLwwlGziM3_sTbVGsXqdMvJONZaMyrYWBYbQGc7DqhO1QzMubhaSKVgLwO7xAMjzbqnTfzs8mw_F4gqpEitYF6B4PmDPIVTiFBbGaJc7CzzUd3_8jtJkzI8X5cSZBrecZtnuD8vj-mv7cT5Jcr6BH26fgdJHVz4mdg1QWjkkvyzuZS0Su5qRbByTWDWvbip4DJ67NWvkxcrcxhyAKXL-wfi8HjAsxEj_MzZr1e_0ehhHRoKUJEmd9dHYUCT9RIG6FXXmQTdFqN954jSBpkyvD6Ok1dtNjF9LAR)

#### Watch the video and explore the iPython notebook on recommending songs

## 

If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebook on song recommendation from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in this example.

#### What you will do

## 

Now you are ready! We are going do three tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

1. **Counting unique users: **The method _.unique()_ can be used to select the unique elements in a column of data. In this question, you will compute the number of unique users who have listened to songs by various artists. For example, to find out the number of unique users who listened to songs by 'Kanye West', all you need to do is select the rows of the song data where the artist is 'Kanye West', and then count the number of unique entries in the ‘user_id’ column. Compute the number of unique users for each of these artists: 'Kanye West', 'Foo Fighters', 'Taylor Swift' and 'Lady GaGa'. **_Save these results to answer the quiz at the end._**
2. **Using groupby-aggregate to find the most popular and least popular artist:** each row of _song_data_contains the number of times a user listened to particular song by a particular artist. If we would like to know how many times any song by 'Kanye West' was listened to, we need to select all the rows where ‘artist’=='Kanye West' and sum the ‘listen_count’ column. If we would like to find the most popular artist, we would need to follow this procedure for each artist, which would be very slow. Instead, you will learn about a very important method:

```
.groupby()
```

You can read the [documentation about groupby here](https://eventing.coursera.org/api/redirectStrict/Jd8dMEL-FosgOSbaOoaEX98udP528LeNfXBhEYX4gw93Bdzhu3PAWnTedl98vjDJcfgiporczGb42szID4TPUA.X30ciXDFyS6ozmvGlTclpg.YWEchPsRnRAqvfpAm4pBUoGyym0qFm831sRTt5b90okVUnyHFwuEAzNxn7eHD6wlvmWup8H39dK_MrWM2nS5dUSyuYuXkqIXiLLtZdiNgX-_9kvH7nmQpfZwWn41tExuMf4SNHQXxbIlXmJt02FWEV31Piy_sKP7MMLt0qhCFQcCSldaw-ngx-JgRRLNI8rkSuHC53Eyikzwhx84TjZOVajQnx0IN9wbuARJ3Ykowehj-b21ulmHXKegQnAeoxovp1r7X89tDd4XXm_KcoWpu9kevgtAOGDD8jpZi-yWbxKr4_gTg1Y78F3F1nBGFeAKoAUf3ZH3M-ywJ_lPNHykUa1debCpeLM0L0_fCw1larL166GDbKxcG79Pa9Q8MQwUChfuxfBDRG3etlG47EUsZk48PpRBA019QHt6Y3Z38BuB6vRye-S2TBn1EBOMXnumyFTljwGFHwFaX7nedRoBkmLTRMaRSXJ0RWH5ijnnnzI). The _.groupby_ method computes an aggregate (in our case, the sum of the ‘listen_count’) for each distinct value in a column (in our case, the ‘artist’ column).

Follow these steps to find the most popular artist in the dataset:

- The _.groupby_ method has two important parameters:

i. _key_columns_, which takes the column we want to group, in our case, ‘artist’

ii. _operations_, where we define the aggregation operation we using, in our case, we want to sum over the ‘listen_count’.

- With this in mind, the following command will compute the sum_ listen_count_ for each artist and return an SFrame with the results:

```
song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})
```

the total number of listens for each artist will be stored in _‘total_count’_.

- Sort the resulting SFrame according to the _‘listen_count’_, and find the artist with the most popular and least popular artist in the dataset. **_Save these results to answer the quiz at the end._**

3. **Using groupby-aggregate to find the most recommended songs:** Now that we learned how to use_.groupby() _to compute aggregates for each value in a column, let’s use to find the song that is most recommended by the _personalized_model _model we learned in the iPython notebook above. Follow these steps to find the most recommended song:

- Split the data into 80% training, 20% testing, using _seed=0_, as was done in the iPython notebook above.
- Train an_ item_similarity_recommender_, as done in the iPython notebook, using the training data.
- Next, we are going to make recommendations for the users in the test data, but there are over 200,000 unique users in the test set. Computing recommendations for these many users can be slow in some computers. Thus, we will use only the first 10,000 users only in this question. Using this command to select this subset of users:

```
subset_test_users = test_data['user_id'].unique()[0:10000]
```

- Let’s compute one recommended song for each of these test users. Use this command to compute these recommendations:

```
personalized_model.recommend(subset_test_users,k=1)
```

- Finally, we can use _.groupby() _to find the most recommended song! :) When we used _.groupby() _in the previous question, we summed up the total _‘listen_count’_ for each artist, by setting the parameter SUM in the aggregator:

```
operations={'total_count': graphlab.aggregate.SUM('listen_count')})
```

For this question, we simply want to count how often each song is recommended, so we will use the COUNT aggregator instead of SUM, and store the results in a column we will call ‘count’ by using:

```
operations={'count': graphlab.aggregate.COUNT()})
```

And, since we want to use the song titles as the key to the aggregator instead of of the ‘artist’, we use:

```
key_columns='song'
```

- By sorting the results, you will find out the most recommended song to the first 10,000 users in the test data!_**Save these results to answer the quiz at the end.**_
## Deep features for image retrieval

In this module, we focused on using deep learning to create non-linear features to improve the performance of machine learning. We also saw how transfer learning techniques can be applied to use deep features learned with one dataset to get great performance on a different dataset. We also built an iPython notebooks for both image retrieval and image classification tasks on real datasets.

In this assignment, we are going to build new image retrieval models and explore their results on different parts of our image dataset. These techniques will be used at the core of the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

#### Learning outcomes

- Execute image retrieval code with the iPython notebook
- Use the .sketch_summary() method to view statistics of data
- Load and transform real, image data
- Build image retrieval models using nearest neighbor search and deep features
- Compare the results of various image retrieval models
- Use the .apply() and .sum() methods on SFrames to compute functions of the data.

#### Resources you will need

- Make sure you have downloaded and installed Python, iPython notebook and GraphLab Create. [You can find the instructions here](https://eventing.coursera.org/api/redirectStrict/Kl84NwzEd7-Lvztw3Z60QONH-t0faKjM4rqdMqNzhtxDRM7a66XTEAnAnNHXFJlcCEkjqPYfb5FXodUkXlxujw.U4tQ3pgGbDeHMviww524QQ.0gbm_WJZwbsxgOv8_ldsFERkJyBD60bcXd33W2lI29muIen9PZ6Cj6ua-yTGJEcKYANw9JtvKwLHv9cowZwqmsvx693Jq-p6yayq3f-_iCdTDmOjHQrEW0t9ibztA9fWQAiLiSyGwrLF9nz-v_HJvpBNpv0ikfzzL2HW3K_KwU2AAxVJiSu1BoJXAl2aAI7Tg52IWD-7ccoKuKUy1LcPBTQA5zXf7fpj5naN174rLsinFdUogUrd0NkW8NWOAes8V9S9cL2qeGu29dOz9r14HqP0rL89cnNPLzlQJrj-Mwo). _(If you are using an ML package other than GraphLab Create, please see the note below.)_
- There are many Python resources available online. [Here is a good place for documentation](https://eventing.coursera.org/api/redirectStrict/Xy5mAWJ39SsUinyatKowjpILNfP2YVx_qr_RcevBGWPlRHdER8wAby5Y-inolYFMwukQoB1tbNc1GHUzfWgzzw.-mv_jSLH4AjaMxThUJYYeA.xbqaMnZ7ihLBtkl5p30j1DzeNDm9XxabvUIHjsJ5-9vZgLgzVfQWefnunt0oMiLCppsUOMqvzenL7_ju1VW6tjUqp9x8esxzbD-tPPM2tl9oOsVNyBKV0KtwL8iPhR2by5y6UZdPpcdd_pVCcWOqIi5bpq-JJHfAAHqxNBTfCmlec6RQjZQe0_8H8L55IRxue4V2ndZktbMclPJ33Bv05zFSS4dS-_-8YGdOH5-q3EFurHKxmywdGzij3Zdkvg64kwLtL7d0Jd4MtRdl92pnLTxjLKjY6J262xyOV2Mi-tiaCA1UezKZoS2IP3byb4sa).
- For GraphLab Create, there is also a lot of information available online. Here are some starting points.

 Learning Concepts about the Tools [https://dato.com/learn/](https://eventing.coursera.org/api/redirectStrict/mz8Zk2GeAl1D4Sh3ug_VO2m-FAghfrM1lMHRe76ZFTLiKtzusFzAsLr6vU34u65_AcNjgXjXScuQFp2IBYfoCQ.hF7da8Qpl58lYN_dvJE-Nw.ZqrX3T7j4i4eSNrefdW1a2Lvwc1JX6Vv5EDNIuRpegyEs6EbHzNY7fhJpFzGf5TWfAiUtuAldcF3LN0_kXPd_q2jvzAAmyIAg71epvLVxO4gNjGCeA8yaeSn_6LUKbSd0N9BkFg5cIQHGcm3puXUzjWqwv4asAbLBAOzmBqM70xhknQW5SGLMPNW6aSOfDG2N09vWru1g1mdmcQuQZ6QOoeoEa9a76QqcW6edLwVKi4lkpjM4yi12KBnqnori0ifMw1hoOAKhkY0SCOEXe8n8Q)

 The User Guide [https://dato.com/learn/userguide/](https://eventing.coursera.org/api/redirectStrict/w7Gmce45fjEsAPQPE4r5x1L0QHLENRqBFroWovxnDha0fdWOaws_uYJ5segQgb78UbBlluTbszC_ZbmL1e5g8Q.D-d8r3WSn21QCVUPvgypqA.W9tfbOoEOLDZsYp-QhkuqmpFvNEYLh6yBMDrnBF5kvVpumoj3MBo9Twkibq6gs_snlToOxmAg8-htoPKrDKryujem1OBhnkxv3rL9hipCAjGuASoBdFXTN55Dod5Dqp-9FMpXbkkTeFMt7FvwQkZMuxL3hgKGbKvjCxL7DBmzo3WLBZIyU9dD5I6L_WIGZQMrkbgXrHA9_NcwwKyKWLxZCPhN5fsPgQXaLyrSia0x1eJ4ii3jo6bqcY7eOM48jOA4bVuyYmswyjAHbnhMGQt3RXFRN60VdXwIUt0Rdm_Yjo)

 More Detailed API Docs [https://dato.com/products/create/docs/](https://eventing.coursera.org/api/redirectStrict/Oyj3sUdoaZvBi9cLsU5W09XZretMqFDvj8R4YABs18AiiYrIdKUMaQ037q0rP5ON_jT6orN8iYhNcjvL9legXQ.YpdaRr0vjKQG9xR2PwHD-A.MCbVU0X0zsAA3IAFAttAWYPeQxkJ-lxXlYDAUo4TYbaniI6TGHv7oogR1hS1g6Pq9rykH2a0flEbLT8LrM2TZytz1Tolla3Hc13xiX9IbSVbdv4OwPJNd3TLN9mirMowatMFAHnLlcpIKK6qoTDxTt70sEBTBJX3iNkYfgHUAgIUZm7_xi9n5QJoOpnZ0ztFHiPWa9LYLDKYK0Wqq_6W_Ftq_AHCP-QDOejAgrWnqzlNdCV2-7aZmf3ydrdhZAVkBY_ESH0NuBtemr8VxAQrpXUEPtD2LrAHnz8mRkAzZ4zfaWudA03evW_Qy0H1uTDt)

#### Download the data and starter code

Before getting started, you will need to download the dataset and the starter iPython notebook that we used in the module.

- Download the wikipedia dataset with training images here in SFrame format: **[image_train_data.zip](https://eventing.coursera.org/api/redirectStrict/porNnJRM3E54VZFqQieDRDmNSpJKfSCkyEb9PN36ueUWMGbFYrkxS2SXTCzHtEfJFa5QD886-Or5gdBRR9I5eg.FPuhxycxXBeweFA_udyDww.ZM7_m4Z1SPaKkOwPSUIoJX2hb0t0nTmTmJIXH5tyeOq7egpJwXuV7sM71kWOGJBl2BPh2CRPxlo7rmeL91wVYjzc15L9HZcua_AW7FiyPGpMVCYbtpfzgVFnQN5sCFrnU0tcTFKKRDE85zi93DHbLHFXGwNwI_ms3SW70sEidEBGep_MRyfruy-H4SlQ9qwbKft0lNsG22jaONxRFbIAfTazxecNHEyTIvbyixMUC4zP0NWM41tP8S1jzWawZCg9B4lmFfvoA_VPBjpRQhe8smme37T5QpUM-VzLSFEHm8-Qq7lDlM4OY-8ArDZE9-b6T8KV1eUGEh-ETjzW5NOah9DiCuifjHVe6qJp0cygqparTWINHpeODYCOCMw_2du46gRUOumyDQVl7SY5lMRg557f5AiMloHHCP6X1l3TtHfO1eochcbYBVmtfe0owXnypgbDp1FBA_F1-fibLowoSg)**
- Download the wikipedia dataset with test images herein SFrame format: **[image_test_data.zip](https://eventing.coursera.org/api/redirectStrict/rKd30MDyaxg5mjYvgfac2yU6SU22XEg97bjo8bQ8SDe1JdqtGGDOhviuKHSt3EMl6u4dwxe-_-kMWMSYZSc5kQ.wiUXo2oClVMaxeruTNcyEA.RLPxwcWNBsN_4IgQG7q5fp-nOiLirCb9-L0anKEE4lLgzIsgokm7lab85z15Pdedeu9S53x9YfiLUx6YZGhNyCqj1PXU7ALWKIswVh4M9al3gxc74UB7eLEZMqHdVwvK9q6gJ8AaAv3oxEwxjOHxhwB2kQMmT1vYMtllefYsDtHwFqsPr6oBbgPM-cPiW1dPKEWjtcLccqfuvzHYDF2dvoJ9NZ50jgxCJB5dlFsVlWPB2ZmvNRo5iQmjXGkd__dzn3SlnfV3bpUVOxY9cDFEzBNzJLJ08axR6g8mqHP9Z47XRKbCDnyYYwEInD35mom7RgNiFlNxl2ukqIv7mVndPPLeayQ0dgHtrtZBQQG8KKHTs42_GWGIp8fh5C1jRC_B-QkBAR2v6pTwcPBdbxK6v92qFoeGDdUU2CnwgoI-ocNz-w_XkprD_05t03sh5RkGKTRrdN-tJFXJb5VS_V8wQw)**
- Download the image retrieval notebook from the module here: [Deep Features for Image Classification.ipynb](https://eventing.coursera.org/api/redirectStrict/vz-M8ws1qiXKB1NbgTGk7YCCGjSO3CfRj3CZ_1U_eZwdenY_Ywmb0sVsdGU2yD-p1VDYNjWCeNlbTX_RoIJgog.5jwC1GPegMxEyrVRLoqsRQ.ux7rwbrgS5JKasVpg7FbYYyGz-s7aB_0kBANr_2yDPyJ4Ddba_NwHoNk7umIsYIqbbXEiSdb_UadVtrXvaJnpUexO24wSz1o-mC686O1IzC0jf5VCmFtsdDZLnr8FrKDh1Vv6Ki7jF76I7nZ-WIv7IjuelYUNmKPbWJvH8XHf2MT11N24MyIyLH4ZKDZAa0GDJbADObHk-ZTmw4-m-U2mV1ORsDYsQ9clNYNWlNRS9gPiNrWy7XX2yelGBSGt6AZGr6SSI7VruIYa4aiMuhdwymV3BssAKWJEUidITm4fpRxYK_moeGr3HDdysCpVmL5ELfFoID7TVZXIWuSDcWwbwMqcVnEvSXS5KAw9MQ9SVEHulshvl5mau5UjuLFfr5mXeLVcjz7qUYLws1K4bsxsEHsg_TAypcI8yuyPsxVOn5AVIIPF43myuYIoERwOfVYXTQdn_fBQKpROlr2cCaRpwkCYn1LH_HNxQkdk3ArjGdzKZHGBBlFCbGTv-9UqbFJ2YXJ5tWYboFflhVVwt8_kMd8dkgBGkKswNnooNxQBCs)
- Download the image retrieval notebook from the module here: [Deep Features for Image Retrieval.ipynb](https://eventing.coursera.org/api/redirectStrict/oEdxRteHScEZuEPM0Bc_3PLPaCM5w2pzqvNv1Z0IxvQG0LTqLAxDxu8bPv4OoSbJOcVYCq8AN4y854y7bzHBlA.lhJ7eD5kLFKPKhP9hEysIQ.AWTvjdtdUuBlxVRWlTFcnBU3SJXy1R-pMTMt93Wb_3GmpLxAVIO8MXEtROjki4A2ie9fJ5X7Y0eroPtX0g_LKnIKenZSu-YfpwR9pCcQl_vxXZB20jY07Y2_9KAMhpH3yu9u6D8XgFCs6ajjJ0mcB0EZ4e1QnWUj5Hp738SbmX1688phhBUmciBIHDUqEwp2pgHele6bnMJWJiMFGAJvUtf4AOPjOOP067FytdI62Li_FXIZGo-_CY2rNSrUhLH04onSZyyXIT7zJX198YANtGqwbqNSwn7hbCnUdUmXxMzSJiDG2vfmi3KEowHLtv62ewj18f6xvrKLU-lJyygzD50aTCu7ve9ryUG26EnqqLnlNtGWjIlRGYlJmZa44ycB_pVA3Mw0SU11uXIKIbVdT2TSdlneLPqyrVMfgnszYFwcdDKpL2OAk3KEy7qp_Au1AgOjAzNYeAFWvx7qctMhO9jwzaqLUJo3IpoO16qJ9isbO6JulA0V-bmPLLnjc_Osl4spiuAgKeQtBw-NKXgpAg)
- Save all of these files in the same directory (where you are calling iPython notebook from) and unzip the data files.

Now you are ready to get started!

#### _**Note: If you would rather use other ML tools...**_

You are welcome to use any ML tool for this course, such as [scikit-learn](https://eventing.coursera.org/api/redirectStrict/J77mSjV4r_3Anm8IIhVxAiByCgipMTTuaRozAapMHT93JYjDdEV70RG9-gzUzbhYqaclI2Uxt3nRU4qeRQpFeQ.48ZVS3T7ViOpPyGl187b7w.5a5KYFANDuJxx7cCoA6zdd9NQh3lVX9Bk3Uy6xM5onYt03f6lB8Rsk-HdytbpBmlB6dNvXhpe3_QKhN-elcviqDEUQdDnIrNQ6M2syXmRY8IFWvPQQV0ulu3ZyNZJzpgjT80ichlYAlya3ExhQPp4jigl7mQ4kjGuQT8hXPOIB0NgmeqL0ucJLfNBTjuv9NPfWKhtr0mdyeZZQrWTpRBq_E7d4RuuaA1XSU8Nx9EvelevZ4Dy_hNJ9OvlhLp1JnbS2LEoymhKX1dhqoCYOb6PwY6cONCAiigksATWnJUDzo). Though, as discussed in the intro module,_ we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)_

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in [open-source under a permissive BSD license](https://eventing.coursera.org/api/redirectStrict/L4f9356-TNOljHL0b0tyg9jtesfih4w-HCiQG8gpOUJ3pMa-DvOYYlArswaNcsgdkQ8k30boetXOoCgNOECeOg.gnjg0TVqyhzp0t4ASfPo4g.scjVZUHJf3e58kdrPojfeFpywt28UzVh3jarFC8SwmfGWS5w6KHJf8F8JdlzTps3B2ONEzApKYpc4Jgt7P4exU4uq1pGD9OEBsrV_4qOWX5fJW9EaLN_KO3GW_jCfjpjQ8f8912zran30_jJpsPzVK-2DHc95KVVDfY-fcfpUYyB0Cn6JEnq9J01fNuWpJsRVy1eo9pwiAUdHpTXL7zGZVqeIWLPmwz1xmKvRO6e_wOEd_7Kv7rATRGYIH3SXMr2kHTFPOBWFNWZlTo0PbnrLLX930839m4zT23Dj09jYhU). So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use [Pandas](https://eventing.coursera.org/api/redirectStrict/OiJA4UMbi5QpQGrXsfQRBXjyj3ru4FWQFXhjkkum2KkSYS0UU3A1R66a1ENe2b2PC6e8PqwMgcvSkCmjKYBntQ.CKgcCNhy66mJ0LU9tpZSow.iedlgWIrH-L_KsgA0qEy4VZmxr4DwIaJrg03qmskm17rIYn_DfRH0HJim5Ko7pdTZ5BGbsh-Elz2szd772OaKjv0ML39SEkC7VWKaGGy0Ycs3orF20yIwT81Wa7_tA1f2w8p03iu2K9itpO9VGwoXEWuODHOsl-XjcOSkRtnGGXRFX-sVuirdYWC1IbI2fTBozlwDlsU_rd2CKy-QMaQxPZ3UKkDynFwitWc6FXCsDdor7GQFH093f2XdCVQngW9C6U81GTDuIYCrDrGCsEIyQ) or other options out there: [image_train_data.csv](https://eventing.coursera.org/api/redirectStrict/CFM67RmHii8HoUA-9JKQaox4k2T-U6SU26pQHB5pxHXFToibjVRhxNNnfHE6GXuyyu4bT0T8luZHz6itiOjP5g._SdW_-ZxWehXnsYP7ddTSw.v4uVCnrG9Nh8QMAXrvi2vHwmhD8G2uShvB_puKD89qQUyTCsSwJ5QFZmrXNkgTvskCqbub8kT9LbZkyA07T1YwxdcBMeu-sV07DV0QdR1vw6ehIYzsZp56AD03MEA3CE2qZMOWxUu9C8QfW24csxU3Sl3oiB_hJ0j5B4_S0wcltha2JNznKbZ0nqPbULhhFJs25c_cDoa3Q1hq8-AZVbqK1H8Tg8GTVI5up9u9YjSqQ651YbCsHAIHWfl-Ydelj7SOdQqvrq630NeczQbFVOQ2LEovULtRm8j8W925lknwjNJ7PSiu_biNLTMYAB1CeebYMstjhFdcYz-PGev125OhFZZ3czMPsiy7q4w1EmVUe-pDYex8Am6aY5HJ3nIGPgmEv7ivTPabGonT0rbiNXoA) and [image_test_data.csv](https://eventing.coursera.org/api/redirectStrict/0RkNUslTAeug2Hc0_Hh8eqNjICn6Fas7Sumnx83TR1I3HubM4QAxu211jcMduke_HkHSAw3ur_-9HtzOBvbrsg._ex1-Ikw3NykK9Os1CsJBg.ulFVaVHl99x65Ss7qyrHAx6XV9-osWw1MwVu4XUS-a41l6X5wxPn8TT_GutqXv7G8vxJfLaoCsAtJi2xgKQiy_9xsO0kv9xrelEh26_O6QS16hZFtW65jrso0nljBwsSqyc6Xv9VBAk_CJapHoGGKYtmaPh2QKaKfqJRPWZuxZf07w-fuhamBXWJzy0Mj4FMRmsiODutqfYA5PEgbsdrDoZir_B0fo5vON5ZRU2oRp1hBmNymPjJ7SoDuw6iMIjiHX8WTpq3qs1Bd45fcZX6r-LoTxNUgYEmULlDdXD13Av3-Rz6u6CXrDFyN8mnNc60GjmKDumnaqv739_S-HSilz_l-Up9VNLg-jMMfd-sC_cdSid2BaL7MPRefzHsL3gi7hsNmB4cldhcKXAWsG8aVw)

#### Watch the videos and explore the iPython notebooks on using deep features for image classification and retrieval

If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebooks from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in these examples.

#### What you will do

Now you are ready! We are going do four tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

**1. Computing summary statistics of the data:** Sketch summaries are techniques for computing summary statistics of data very quickly. In GraphLab Create, SFrames and SArrays include a method:

```
.sketch_summary()
```

which computes such summary statistics. Using the training data, compute the sketch summary of the ‘label’ column and interpret the results. What’s the least common category in the training data? **_Save this result to answer the quiz at the end._**

**2.  Creating category-specific image retrieval models:** In most retrieval tasks, the data we have is unlabeled, thus we call these unsupervised learning problems. However, we have labels in this image dataset, and will use these to create one model for each of the 4 image categories, {‘dog’,’cat’,’automobile’,bird’}. To start, follow these steps:

- Split the SFrame with the training data into 4 different SFrames. Each of these will contain data for 1 of the 4 categories above. _Hint: if you use a logical filter to select the rows where the ‘label’ column equals ‘dog’, you can create an SFrame with only the data for images labeled ‘dog’._
- Similarly to the image retrieval notebook you downloaded, you are going to create a nearest neighbor model using the 'deep_features' as the features, but this time create one such model for each category, using the training_data. _You can call the model with the ‘dog’ data the dog_model, the one with the ‘cat’ data the cat_model, as so on._

You now have a nearest neighbors model that can find the nearest ‘dog’ to any image you give it, the dog_model; one that can find the nearest ‘cat’, the cat_model; and so on.

Using these models, answer the following questions. The cat image below is the first in the _test data_:

You can access this image, similarly to what we did in the iPython notebooks above, with this command:

image_test[0:1]

- What is the nearest ‘cat’ labeled image in the training data to the cat image above (the first image in the test data)? **_Save this result._**

_Hint: When you query your nearest neighbors model, it will return a SFrame that looks something like this:_

query_label | reference_label | distance | rank
------------ | ------------- | ------------- | -------------
 0 | 34 | 42.9886641167 | 1
 0 | 45 | 43.8444904098 | 2
 0 | 251 | 44.2634660468 | 3
 0 | 141 | 44.377719559 | 4

_[To understand each column in this table, see this documentation.](https://eventing.coursera.org/api/redirectStrict/X1956MuRpVcdkHY0s7X2x8jyD17X0XgXQXVnm5jcvPcv6-YlH9IS9ainG1ODhJwl0PiM9d5uTwE52leET7hqtw.AhfgFVnFcavdb-HLNI7uVA.LiCdELSoyjAxrewHLta99I4c4G-ImLDbvb0XkO2j8FOXQX3M_kR-rmhj93Mnl546EmZpfW4Q3_ylNNEPSr68mSuEM_6vaxkxDKWCGpWz-afIZP6eetRzEFb0LMxiOE033sSp_bQykUw2ohC1kflxANtrpuwCkfQCIwEgxr3rlICRYtlnxVJ0eNrUS8Q0HCdtnXmFPD-kEB34g9uAxwnnxx0UdYovkaIWleyjsRKNOjme1ZfL9kOQ8fuYkGGA-lTeXfAS4JhIhDDt0IfcsYTMAIYGubHma6nPvQQQNeIz-GSWjKO-pS8rNDm4IfexbT0plTagVOWtJ1o1wo7QOjANwC7TDINkbE_pA6j7zRhYDENgDrYaq4LVBxlnduL8_OKotVi0UHItPhlJC0QtXg0KyA) For this question, the ‘reference_label’ column will be important, since it provides the index of the nearest neighbors in the dataset used to train it. (In this case, the subset of the training data labeled ‘cat’.)_

- What is the nearest ‘dog’ labeled image in the training data to the cat image above (the first image in the test data)? **_Save this result._**

**3. A simple example of nearest-neighbors classification:** When we queried a nearest neighbors model, the ‘distance’ column in the table above shows the computed distance between the input and each of the retrieved neighbors. In this question, you will use these distances to perform a classification task, using the idea of a nearest-neighbors classifier.

- For the first image in the test data (image_test[0:1]), which we used above, compute the mean distance between this image at its 5 nearest neighbors that were labeled_** ‘cat’**_ in the training data (similarly to what you did in the previous question). _**Save this result.**_
- Similarly, for the first image in the test data (image_test[0:1]), which we used above, compute the mean distance between this image at its 5 nearest neighbors that were labeled **‘dog’ **in the training data (similarly to what you did in the previous question). **_Save this result._**
- On average, is the first image in the test data closer to its 5 nearest neighbors in the ‘cat’ data or in the ‘dog’ data? (In a later course, we will see that this is an example of what is called a k-nearest neighbors classifier, where we use the label of neighboring points to predict the label of a test point.)

**4. [Challenging Question] Computing nearest neighbors accuracy using SFrame operations: ** A nearest neighbor classifier predicts the label of a point as the most common label of its nearest neighbors. In this question, we will measure the accuracy of a 1-nearest-neighbor classifier, i.e., predict the output as the label of the nearest neighbor in the training data. Although there are simpler ways of computing this result, we will go step-by-step here to introduce you to more concepts in nearest neighbors and SFrames, which will be useful later in this Specialization.

- **Training models: ** For this question, you will need the nearest neighbors models you learned above on the training data, i.e., the dog_model, cat_model, automobile_model and bird_model.
- **Spliting test data by label:** Above, you split the train data SFrame into one SFrame for images labeled ‘dog’, another for those labeled ‘cat’, etc. Now, do the same for the test data. You can call the resulting SFrames

```
image_test_cat, image_test_dog, image_test_bird, image_test_automobile
```

- **Finding nearest neighbors in the training set for each part of the test set:** Thus far, we have queried, e.g.,

```
dog_model.query()
```

our nearest neighbors models with a single image as the input, but you can actually query with a whole set of data, and it will find the nearest neighbors for each data point. Note that the input index will be stored in the ‘query_label’ column of the output SFrame.

Using this knowledge find the closest neighbor in to the dog test data using each of the trained models, e.g.,

```
dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
```

finds 1 neighbor (that’s what k=1 does) to the dog test images (_image_test_dog_) in the cat portion of the training data (used to train the_ cat_model_).

Now, do this for every combination of the labels in the training and test data.

- **Create an SFrame with the distances from ‘dog’ test examples to the respective nearest neighbors in each class in the training data: **The ‘distance’ column in _dog_cat_neighbors_ above contains the distance between each ‘dog’ image in the test set and its nearest ‘cat’ image in the_ training set_. The question we want to answer is how many of the test set ‘dog’ images are closer to a ‘dog’ in the training set than to a ‘cat’, ‘automobile’ or ‘bird’. So, next we will create an SFrame containing just these distances per data point. The goal is to create an SFrame called _dog_distances_ with 4 columns:

i. _dog_distances[‘dog-dog’]_ ---- storing _dog_dog_neighbors[‘distance’]_

ii. _dog_distances[‘dog-cat’]_ ---- storing _dog_cat_neighbors[‘distance’]_

iii. _dog_distances[‘dog-automobile’] _---- storing _dog_automobile_neighbors[‘distance’]_

iv. _dog_distances[‘dog-bird’]_ ---- storing _dog_bird_neighbors[‘distance’]_

_Hint: You can create a new SFrame from the columns of other SFrames by creating a dictionary with the new columns, as shown in this example:_

```
new_sframe = graphlab.SFrame({‘foo’: other_sframe[‘foo’],‘bar’: some_other_sframe[‘bar’]})
```

The resulting SFrame will look something like this:

 dog-automobile | dog-bird | dog-cat | dog-dog
------------ | ------------- | ------------- | -------------
 41.9579761457 | 41.7538647304 | 36.4196077068 | 33.4773590373
 46.0021331807 | 41.3382958925 | 38.8353268874 | 32.8458495684
 42.9462290692 | 38.6157590853 | 36.9763410854 | 35.0397073189

- **Computing the number of correct predictions using 1-nearest neighbors for the dog class:** Now that you have created the SFrame _dog_distances_, you will learn to use the method

```
.apply()
```

on this SFrame to iterate line by line and compute the number of ‘dog’ test examples where the distance to the nearest ‘dog’ was lower than that to the other classes. You will do this in three steps:

i. Consider one row of the SFrame dog_distances. Let’s call this variable row. You can access each distance by calling, for example,

```
row[‘dog_cat’]
```

which, in example table above, will have value equal to _36.4196077068_ for the first row.

Create a function starting with

```
def is_dog_correct(row):
```

which returns 1 if the value for _row[‘dog_dog’]_ is lower than that of the other columns, and 0 otherwise. That is, returns 1 if this row is correctly classified by 1-nearest neighbors, and 0 otherwise.

ii. Using the function _is_dog_correct(row)_, you can check if 1 row is correctly classified. Now, you want to count how many rows are correctly classified. You could do a for loop iterating through each row and applying the function _is_dog_correct(row)_. This method will be really slow, because the SFrame is not optimized for this type of operation.

Instead, we will use the_ .apply() _method to iterate the function is_dog_correct for each row of the SFrame.[Read about using the _.apply() _method here.](https://eventing.coursera.org/api/redirectStrict/mdSNFZ39vpiX4WklE3lM1dnk_0GgiGiGRps_npFMsqvzk_pQX2qLTKt3qeYwH0Gppm_OtDqlsVskR74JQYTUkw.PXl6o3volcqZ2yNTSbuxgQ._oUlSICgR8aYHF1NiKtVOn0p2hpyyueoOlio_ayXiHgb7x22MOqYAzUE6pbmbHNnl4oTJotXb9UFU-mtganBfqAul2XiyriWeMLNLBXLp1aoBYa4uVJa44lcS23bq2TCEwCc5i6NuhQPT5hDWW5O99Pbd8MHDL3oPysFD1IhgSeFM7VNyVwPbHckD0PnGa_p6qVXitkeQD30vIMXJSsgWhr4rfqfHqONULHRdT5hLf88FqcwdyiiPebI26_cS_4oybR7BeXRe9Z5-1zNisNhhSeGnFuMuhAZBbDG7udFNILFXR9xZHOsuW2ZJaAxxmdK6FvAKKzF4NsUlNkaKAnFremwNUiVEGW-cNgfJSD0I5w)

iii. **Computing the number of correct predictions for ‘dog’: **You can now call:

```
dog_distances.apply(is_dog_correct)
```

which will return an SArray (a column of data) with a 1 for every correct row and a 0 for every incorrect one. You can call:

```
.sum()
```

on the result to get the total number of correctly classified ‘dog’ images in the test set!

_Hint: To make sure your code is working correctly, if you were to do steps d) and e) in this question to count the number of correctly classified ‘cat’ images in the test data, instead of ‘dog’, the result would be 548._

- **Accuracy of predicting dog in the test data: ** Using the work you did in this question, what is the accuracy of the 1-nearest neighbor classifier at classifying ‘dog’ images from the test set? **_Save this result to answer the quiz at the end._**
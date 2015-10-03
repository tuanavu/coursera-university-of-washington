## Retrieving Wikipedia articles

In this module, we focused on using nearest neighbors and clustering to retrieve documents that interest users, by analyzing their text. We explored two document representations: word counts and TF-IDF. We also built an iPython notebook for retrieving articles from Wikipedia about famous people.

In this assignment, we are going to dig deeper into this application, explore the retrieval results for various famous people, and familiarize ourselves with the code needed to build a retrieval system. These techniques will be key to building the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, **_instead of uploading your code, you will answer a series of quiz questions_** (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

#### **Learning outcomes**

- Execute document retrieval code with the iPython notebook
- Load and transform real, text data
- Compare results with word counts and TF-IDF
- Set the distance function in the retrieval
- Build a document retrieval model using nearest neighbor search

#### **Resources you will need**

- Make sure you have downloaded and installed Python, iPython notebook and GraphLab Create. [You can find the instructions here](https://eventing.coursera.org/api/redirectStrict/GcisOGuMlUuHxHnOJT7Uymj1xy3E8lJCtbVsp7z3DS3HqQI-6363aiXO6oeYIs9b-JCx-3mALcZWNjA4JIcksw.A0uAR3GuoQXmeVALBrDI0A.iqzUGJhqNzlEvqOO8s1vLtGxYslF6DmqYp62noMaIYRTSV0zLD-mjDDCzYMhzwWW6-kiCDx0Fef02_5e910dOlk7v4TO_aWCuhXgrenCHlhzeypL-MbAG83z5ohXjcLVCTUDfe8i4Q_c6X7-Ma74kmyNZK4u1yrTosmuzkQ-el0xlsUc4OFU4g-QD9BLSp0NgfTuTUAWXnePSiFCfKOO8XMudrxYVjNgJFdFDVQ9dZwTFu1Rsr4ilxx3BdkCJ8K0e9NbkTaC61-aZE2Bmns-0csynm2-0JAZvQAgzvjfJmU). _(If you are using an ML package other than GraphLab Create, please see the note below.)_
- There are many Python resources available online. [Here is a good place for documentation](https://eventing.coursera.org/api/redirectStrict/Pd5dN7A7A5xEby8hM1ldx6mxIIjqWk4JT_7KMZ1lit-kURdEc7GHPlhRKkEqNMqpuSOUpce9FjIpfocZsja9lg.T46On5GxTO5h5Jx_NWaZjQ.ywuDqNw_xZhQQF4qB9MzDkcPnqgDxFFVTEy51GyXZwr7nX6uw1RXKfEWzOpVKBPhAHHbIz08hFNfiKndMSOGTxdxfojwTkwKt04L6JFj7NArtobdTt8KVHaqcml9xvW8y4__Sepjvl5_L72TXNqnO5nc77f1Ge23VLC94wO89sgKCoFv4MDCyRvBrmlztsl8aXSrEqRYCNX5rkIjEw54axaJ_2hHJlVfewsYu4dhJrIUFlWYImb87qayVNXuVL7UgQqqDHdMIRR0kEeWJlNHRSgP_lvLUIAe3BrgMq0PkVNQTllTno13gsFD4f_HGt0W).
- For GraphLab Create, there is also a lot of information available online. Here are some starting points.

 Learning Concepts about the Tools [https://dato.com/learn/](https://eventing.coursera.org/api/redirectStrict/Lt6aM37Npqa5kZ1QwPExd8IyJ8A0wLXoACmKUeZIvn61Sg8iWD-tMgppXKvXjIADNDxqclenEdyKuFyuLNOcfg._LePR5kVsB9AqVPzrQCfRg.LlgkSPjt4cZH-A8HwjncQU0VAQg6ISmu7IO21Iv1vYY497xdvoDPPuDB0gvx4bVV5jIM8sOpJyti9M3B_TXUc9nkvq2t3pxSI5ygTxcUULm4R-p5ONWnQs3WBzQRzdJb1xovTDDJ9xgtovD1GeLSKmeO_XI0K3bBLqbK_GtN0FBOq0qTPLPFC1NfkNZDbM4DpAyl2vPJJ3VHEbuKUodJfKAi6LTp70_milZXtsOgIhF2QcipLBB18kDci4tROpLeqARD_AqzWDhYu6fvmW7WrA)

 The User Guide [https://dato.com/learn/userguide/](https://eventing.coursera.org/api/redirectStrict/zN1clf0M6ZfZ1hbt_Q2m8j4Ihh74uxTkSdr6HU2_ag1fFgzH8Hj77cl4GpJVBjUOxT6Ujkgf-zMIk9ev5KuPpQ.FkmF96CiubDCVenLBOMUfw.VMevDO8vKjfMngqCHAzCPtfI6_k5xzNG4YUzPZEloUZLUxPaxsuemf73trVGuxFU_aCIRQdyWa6C38hCvu1Py5r4dx-WAkGPyoop4TraIZp5aHx_CEEwVz_0rwohuNOzoAgW4kRxc2LQ8eqGFzU8F1G51aPenKy7JDDlRwiqgfZ2nIB2lr3o-cYbb5lvp5u0RUjS5pCwv9d5fnAN1d7OxZpzvGx7C4wYp9cp3YqtWV89_2tccgk_aHWxMIa5skYpnyLiQcqtmPfbTqG4_P7SEB_j09cU1sZ3TJJRt96cBZw)

 More Detailed API Docs [https://dato.com/products/create/docs/](https://eventing.coursera.org/api/redirectStrict/n2D9XSHMw2EDEAEV2qKCXWGNftFvlqSO1A2LXv2-dRpLqW8liaeMWf945P_MVdWYbeitXT03zYN7DP2jNH8KpQ.7qet8d-VvOGF4BPhg4Ccyw.21DbBvper_jwbH5qkU6oOk6PCwUsxmXDTXma3KLD9IYsNztJd4bWmP6D13ZVpRV_w8i5HZ4gwQni6XNaj_BgDmz-26wKfO0Uh2Nbk065lv38EF1cQlcYsFbLK6zb4pCey-_Lhvq0e034xclUwE9b6AgkRqktvBYjZP2Zw9dp4FINzNu211102Yzg_qa-f0TgEXWbUq58eqWz1UYhPf_p4-57sk5DPiDUbzS08Tryypf-K1dRxuuh8A-BOcbgOJD5YYOP8PQd0PgepPcByuVDUnv0doIhsCvAbu1QQSM6b1s71-L7Ym1kTbFnG2I0-Ogk)

#### Download the data and starter code

Before getting started, you will need to download the dataset and the starter iPython notebook that we used in the module.

- Download the wikipedia dataset with articles on famous people here in SFrame format: **[people_wiki.gl.zip](https://eventing.coursera.org/api/redirectStrict/GKsSRrYkoeGfc5XjAKr6ncJ3J2ubnIm999yQqv35HDem2Ta9G7KHR0vHwNYFo0f-Z6Vmdknu3Zu3bUKLjLaN0g.dQ7WrvOu097_hXA-L_GwEQ.jyciXHStsYv3slLocnd8PMgNlMfQWxmgbAG7CjqHwZynJhmlBupunsBPgk-m9M5gtckqHwtnp2xIuaeanrjL3wCNfhycxlTM4wSZ4paOra3d-e2T9AYkf1jHWbcm07PeSsoRpDGFCc2sMLE12rGGiGh8lb5VTCsXWLhrOlxJI-y2A5xRb_TfJIyNsYFZLqGohOxhZfZ4cTuQUcWVYNkUyfXoza0Sfy4Oh8uN56yKkyggBgWBvh75PKH_OQQBYQr_hq-Ktl_KHyfJWYhLZTos9J1vI0ZWyc2SMih3Giemggw7un5IsVJfzQKXQGc_LHRuCFcTj9qx87eY_v1dD-j9ZFmuDwJH7hqnkPdGKFJ5dLWH8yqlATycr-Hg6VneVoy_07fpLFLomtWuNxpJLjTsTmWev8hmLtwzDUIyY55Ik1TJkty7b4xuj4smVD7pWEE9)**
- Download the document retrieval notebook from the module here: **[Document retrieval.ipynb](https://eventing.coursera.org/api/redirectStrict/_eqO9XZ27B170KFo6clxBJ4oi_yUvcMdqdekzCHFeafU_qeMrfxrQ60VWa0qrt-n4mfHhY2fzLK2sgG9Za7ynA.YZZBch-KNR48GrGoeaY5Kg.FiFl2enCXjdviicIHf89_Hzi99nT8sWn8DbvAf5onC4LPyTKlB8PpUYW-gLgT8JYHe9PeULpoA_OcjOEX14E3Y9ojALKbUIQ2D1_816EstaQn-ceebdzxnpqGo6Is07Hskmf-YpQTMhG4miRnyOH9e-6Fev842APTR7ruYgihl0S1PYvS2wDYKo8sxxGWGsVtTsVrS20O4ChC6L_-evQKu9pcWT0QeP6nLzQpaC7I41_UGgy_5pjbxaXu2ePbeooQ_R4chYpR65iDGp-nbY9tYwU_4KubldVY8C9rhqXU6n91OG4wjRzTxG9ViuFTmGtto2Z-d5VPGLww-yhJf0b0U94KEiKBEPqUBYRpRujfrmivgJWZwM-E2dvZ88V7H8mIaop5xwB46vN189WWiyhGncNgZyR29izgBa_fQlsQjnXT6qoAlDZEq8L_S6EvTcB28A6fNoFHHRKnBus5r5yuw)**
- Save both of these files in the same directory (where you are calling iPython notebook from) and unzip the data file.

Now you are ready to get started!

#### _**Note: If you would rather use other ML tools...**_

You are welcome to use any ML tool for this course, such as [scikit-learn](https://eventing.coursera.org/api/redirectStrict/Sy3lss2nzwqCTPV1YCcIc_vVuMlpW8mIUPS_WbzTSEBNnKRa_aXtolMLxtyOR4ThjLjjQkjABHzhMlLclen18Q.mHqMnv0rhlSNNRKKZWBKOQ.MC8XuZQoGwCrz8jDaqlYYgBTkS6c27NMgJwnTRbkD-NKLd8zHh7YIKbAD_VfKfNnjZecQhcYYl1cXD5RdOHAaKJhHS8W7kHGIkwAVfWVApmQrfTRmu-s9weDCo3A-6adZYe_gRj8df-cp77X2aIPrbAJdEX79xaBMbblA5ICxiUABrQuu1IveIRi8gaK5_vheQmaIXixYHJgG2bByhgYweEI4j_FW3kJe_qJhPP3cLDOJIHcb2I3W_HOo6wwhGn7oxMpg2hRTLKUxVqM36xeksPPU41VW3DkSX0mniZG2t8). Though, as discussed in the intro module,_ we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)_

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in [open-source under a permissive BSD license](https://eventing.coursera.org/api/redirectStrict/FWvrFKHqSccB0PmlC6vdOGxZX-IgOkGFrIBHNQFv8YL7n16pjIIaBCfXE6N6o3J-GVRIPwXWGd-jokeZ82t-OA.9ZbPB_0u0lyAEtiI-eHidQ.0V-cV7DuFtyUvkTVGml6RQhnduu7Yrvu4W-wsAJIin6GILZbfsTGonXYdQOY7VtBeGso9_qJBaogTnc3vnRSU39kM6DFJ6mWwD4uaERda62Kgp8zn-QT-055c88eWtuprX008Bk2ayLOlKm27H60ec8Ldb_JZXtX4nArPjAdFSlMOle7qzmbbAtr08XkL1XMddq7Be2c8AHkcUTEMHttQSCxMJzzhZPm7udqXpO6FxYJ1xu_Ulj5cDVgv01W2cUqad59EQpu4PpliuYFcBPR6ALzkYjDcGqcvjmE3ymvIu8). So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use [Pandas](https://eventing.coursera.org/api/redirectStrict/L42mCO45hMc2hIHM-3XQ9AbBJfwrnnzOdonEn4sL7VXn-ETKcUERMyN5Wrsu3oo8bhpovrhFQWVWWRhf5cttXw.F9gYl7lkijdr-CNDbFRfsA.TwaXDMp-IVVSvRPM-hM1VM7o851XgAEhO5S_SNGrdlEHQ0J-3OYFofXEWerf3leOTKz2smQ6J1D4sZb7SOrpyHmL7BMak8amhPCC3e77l7LhqvDhCdfjOXAepc3TKt944J5Utd6yAoDZMax8z4UmWJHXb8yiRjDVHen7u1GmSLJ8jOdf2O2y4pNjEMjT4Qmz-CEpXm6sZDvOCL2MkcXmXTHvNfoE1ak5WTnPPEJjNZIMJk7Eo6IqlTbYJ7oNSFXWv7-nvURg_-qr1DtvsnwRYQ) or other options out there: [people_wiki.csv](https://eventing.coursera.org/api/redirectStrict/9gqESG8_D0ZzBDBEgZUDRKrxRmUOchOhQc7mD4qHr1SOWgYdqKQ0baKAlgEH1EbTmJ-sYqyRQVO7pjVlCcgh2Q.IfVnWSjt7nxd9NNWq95jJA.t6KmRyo8-xCIBB7fxZivRvY-UlsuH-SaG8MI5PaZHI6LVtcrOmOR_MZAQ1g9wkSzPDfZMxTKvj4ECAwE3WUIy39azm-mX4Qwtkq3fCvA2F9MNoV9nUu1RIdPazJALSiV91vjdM4UQM8baLC3rKOkhAT6imCx5rtruuT5Iun7mnOD4ptjCAL8a3tsXUN1ob1CQVBChr2rsjoEpfZRAwUJRGuxwxa5ur4GE1Eo3FrX2amZkWEx4Dx98Sd9uNLl6NIRl0-hOmRbcWZMFLbbfPBG5uv75pucgVRvFDqYFTG0lTzjo_aOLyHjXY4FN_JtWrSgluFMOOMV1mdh9ox-lDSwa9tmcifBwhmbT6DjYSFAgIqWHpDXrhJUWTcJebl2WEUf)

#### Watch the video and explore the iPython notebook on retrieving wikipedia articles

If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebook on retrieving documents from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in this example.

#### What you will do

Now you are ready! We are going do three tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

1. **Compare top words according to word counts to TF-IDF:** In the notebook we covered in the module, explored two document representations: word counts and TF-IDF. Now, take a particular famous person, 'Elton John'. What are the 3 words in his articles with highest word counts? What are the 3 words in his articles with highest TF-IDF? These results illustrate why TF-IDF is useful for finding important words. **_Save these results to answer the quiz at the end._**
2. **Measuring distance:** Elton John is a famous singer; let’s compute the distance between his article and those of two other famous singers. In this assignment, you will use the _[cosine distance](https://eventing.coursera.org/api/redirectStrict/qBK8IS3aHLoSfRwTU7HZni0hHvuV76Lci27OC4fsFewxtfi8phWkIRN6jXU83lEyq3CNhFYYHjo1M77FwiK6aQ.jBdADI7M-E5RFjGnHx-XxA.b3j8Q34Ie0pgrnNrHYogxjGUyACmu0GqelMgVwL7LsFnccN2x-1-lgt91VLLkcuHW94i-ys1Y3iLrhjt7JoQAT-LH2WVpW2ZHG8J31suMcqX24kpAvczx3NjgDfdWTQY77KwyKW7cCo8kV7w0G8c-8-PR-KD6mF-E4ZvgvFYkxzHTOqbM5GFXIcJ3I4X3uZpskqq8Ju9kz3DJZ-uO3uUQWBXip2PpV58yzw1wsW_HvL8DHeQYnzr1eUgPKFGHBczW-izZJbH7UAG7DNtfYqdj7q_9p3lhTPztS4f6fShj9FkoYtcLqxjdQUYdfJPY8Kt9tqK7ekYjLFACNyqdTkemXLuOBmS3yAdIdeYmmmm2s84H7KPME6PfMnktEWhtFrkurvuubWntyYeLsz5x6swb0u8ewel2XYY0YU5RGcdKAwrJH64k5PhjFW3tvrz5ta6)_, which one measure of similarity between vectors, similar to the one discussed in the lectures. You can compute this distance using the _graphlab_.distances.cosine function. What’s the cosine distance between the articles on ‘Elton John’ and ‘Victoria Beckham’? What’s the cosine distance between the articles on ‘Elton John’ and Paul McCartney’? Which one of the two is closest to Elton John? Does this result make sense to you? _**Save these results to answer the quiz at the end.**_
3. **Building nearest neighbors models with different input features and setting the distance metric:** In the sample notebook, we built a nearest neighbors model for retrieving articles using TF-IDF as features and using the default setting in the construction of the nearest neighbors model. Now, you will build two nearest neighbors models:

- Using word counts as features
- Using TF-IDF as features

In both of these models, we are going to set the distance function to cosine similarity. Here is how: when you call the function

```python
graphlab.nearest_neighbors.create
```
add the parameter:

```python
distance='cosine'
```

Now we are ready to use our model to retrieve documents. Use these two models to collect the following results:

- What’s the most similar article, other than itself, to the one on ‘Elton John’ using word count features?
- What’s the most similar article, other than itself, to the one on ‘Elton John’ using TF-IDF features?
- What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using word count features?
- What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using TF-IDF features?

_**Save these results to answer the quiz at the end.**_
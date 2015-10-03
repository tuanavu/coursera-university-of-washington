## Analyzing product sentiment

## 

In this module, we focused on classifiers, applying them to analyzing product sentiment, and understanding the types of errors a classifier makes. We also built an exciting IPython notebook for analyzing the sentiment of real product reviews.

In this assignment, we are going to explore this application further, training a sentiment analysis model using a set of key polarizing words, verify the weights learned to each of these words, and compare the results of this simpler classifier with those of the one using all of the words. These techniques will be a core component in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, _**instead of uploading your code, you will answer a series of quiz questions**_ (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

#### Learning outcomes

## 

- Execute sentiment analysis code with the IPython notebook
- Load and transform real, text data
- Using the _.apply() _function to create new columns (features) for our model
- Compare results of two models, one using all words and the other using a subset of the words
- Compare learned models with majority class prediction
- Examine the predictions of a sentiment model
- Build a sentiment analysis model using a classifier

#### **Resources you will need**

## 

- Make sure you have downloaded and installed Python, IPython notebook and GraphLab Create. [You can find the instructions here](https://eventing.coursera.org/api/redirectStrict/GcisOGuMlUuHxHnOJT7Uymj1xy3E8lJCtbVsp7z3DS3HqQI-6363aiXO6oeYIs9b-JCx-3mALcZWNjA4JIcksw.A0uAR3GuoQXmeVALBrDI0A.iqzUGJhqNzlEvqOO8s1vLtGxYslF6DmqYp62noMaIYRTSV0zLD-mjDDCzYMhzwWW6-kiCDx0Fef02_5e910dOlk7v4TO_aWCuhXgrenCHlhzeypL-MbAG83z5ohXjcLVCTUDfe8i4Q_c6X7-Ma74kmyNZK4u1yrTosmuzkQ-el0xlsUc4OFU4g-QD9BLSp0NgfTuTUAWXnePSiFCfKOO8XMudrxYVjNgJFdFDVQ9dZwTFu1Rsr4ilxx3BdkCJ8K0e9NbkTaC61-aZE2Bmns-0csynm2-0JAZvQAgzvjfJmU). _(If you are using an ML package other than GraphLab Create, please see the note below.)_
- There are many Python resources available online. [Here is a good place for documentation](https://eventing.coursera.org/api/redirectStrict/Pd5dN7A7A5xEby8hM1ldx6mxIIjqWk4JT_7KMZ1lit-kURdEc7GHPlhRKkEqNMqpuSOUpce9FjIpfocZsja9lg.T46On5GxTO5h5Jx_NWaZjQ.ywuDqNw_xZhQQF4qB9MzDkcPnqgDxFFVTEy51GyXZwr7nX6uw1RXKfEWzOpVKBPhAHHbIz08hFNfiKndMSOGTxdxfojwTkwKt04L6JFj7NArtobdTt8KVHaqcml9xvW8y4__Sepjvl5_L72TXNqnO5nc77f1Ge23VLC94wO89sgKCoFv4MDCyRvBrmlztsl8aXSrEqRYCNX5rkIjEw54axaJ_2hHJlVfewsYu4dhJrIUFlWYImb87qayVNXuVL7UgQqqDHdMIRR0kEeWJlNHRSgP_lvLUIAe3BrgMq0PkVNQTllTno13gsFD4f_HGt0W).
- For GraphLab Create, there is also a lot of information available online. Here are some starting points.

 Learning Concepts about the Tools [https://dato.com/learn/](https://eventing.coursera.org/api/redirectStrict/Lt6aM37Npqa5kZ1QwPExd8IyJ8A0wLXoACmKUeZIvn61Sg8iWD-tMgppXKvXjIADNDxqclenEdyKuFyuLNOcfg._LePR5kVsB9AqVPzrQCfRg.LlgkSPjt4cZH-A8HwjncQU0VAQg6ISmu7IO21Iv1vYY497xdvoDPPuDB0gvx4bVV5jIM8sOpJyti9M3B_TXUc9nkvq2t3pxSI5ygTxcUULm4R-p5ONWnQs3WBzQRzdJb1xovTDDJ9xgtovD1GeLSKmeO_XI0K3bBLqbK_GtN0FBOq0qTPLPFC1NfkNZDbM4DpAyl2vPJJ3VHEbuKUodJfKAi6LTp70_milZXtsOgIhF2QcipLBB18kDci4tROpLeqARD_AqzWDhYu6fvmW7WrA)

 The User Guide [https://dato.com/learn/userguide/](https://eventing.coursera.org/api/redirectStrict/zN1clf0M6ZfZ1hbt_Q2m8j4Ihh74uxTkSdr6HU2_ag1fFgzH8Hj77cl4GpJVBjUOxT6Ujkgf-zMIk9ev5KuPpQ.FkmF96CiubDCVenLBOMUfw.VMevDO8vKjfMngqCHAzCPtfI6_k5xzNG4YUzPZEloUZLUxPaxsuemf73trVGuxFU_aCIRQdyWa6C38hCvu1Py5r4dx-WAkGPyoop4TraIZp5aHx_CEEwVz_0rwohuNOzoAgW4kRxc2LQ8eqGFzU8F1G51aPenKy7JDDlRwiqgfZ2nIB2lr3o-cYbb5lvp5u0RUjS5pCwv9d5fnAN1d7OxZpzvGx7C4wYp9cp3YqtWV89_2tccgk_aHWxMIa5skYpnyLiQcqtmPfbTqG4_P7SEB_j09cU1sZ3TJJRt96cBZw)

 More Detailed API Docs [https://dato.com/products/create/docs/](https://eventing.coursera.org/api/redirectStrict/n2D9XSHMw2EDEAEV2qKCXWGNftFvlqSO1A2LXv2-dRpLqW8liaeMWf945P_MVdWYbeitXT03zYN7DP2jNH8KpQ.7qet8d-VvOGF4BPhg4Ccyw.21DbBvper_jwbH5qkU6oOk6PCwUsxmXDTXma3KLD9IYsNztJd4bWmP6D13ZVpRV_w8i5HZ4gwQni6XNaj_BgDmz-26wKfO0Uh2Nbk065lv38EF1cQlcYsFbLK6zb4pCey-_Lhvq0e034xclUwE9b6AgkRqktvBYjZP2Zw9dp4FINzNu211102Yzg_qa-f0TgEXWbUq58eqWz1UYhPf_p4-57sk5DPiDUbzS08Tryypf-K1dRxuuh8A-BOcbgOJD5YYOP8PQd0PgepPcByuVDUnv0doIhsCvAbu1QQSM6b1s71-L7Ym1kTbFnG2I0-Ogk)

#### Download the data and starter code

## 

Before getting started, you will need to download the dataset and the starter IPython notebook that we used in the module.

- Download the product review dataset here in SFrame format: [amazon_baby.gl.zip](https://eventing.coursera.org/api/redirectStrict/6hh45DB611nh7GHP-6nC6rrsLhWEos2nNZFZxwCABbXVguqFBvasIb_lYJGNpxKr_q_3iRfeu0acAxUj8NlUwg.c89HKYHOxUVWfE1VuvmZiQ.ptc5L4RlgxUaJ3IN97CuMjipgIlSc7wcd-J4Z5a7wEmrRDLPSI0DcGEFCiDTQd8ZiLqSsxBTsEDdxh3jl8Cqfe4LSltjw4BIwZxUturaqPSsGfqe64VXwxcF1RJWl3PgImnytxts54sPbb0btFjQzrcVrippdB0J0K1rJWDTvVbbkFhb5herH3UPJO3Zj6MNh2-fG6fIMSvkOcy9aDcl5CSuN7Avz_l6-qiMJjVOX-KfXGbcrXncMpHqBvrAd8R6MvSsh9f7WHu_6eMPZME2LbTLfW46c_SCXmyICOjPNzdCE32-Jc45ECBO683dUm_kgjEwgEd2Ce8tTWP353aR2vL8RftQTIqLxc9HI_u1j3tb5C3QrLcMg_p03pQhaTOR-O_bekSXrIJRkLfLZmHTndIigEnkjl0OTwYVCEpSA8Ri1Jo6fX6xYZb43eeC4rL3)
- Download the sentiment analysis notebook from the module here: [Analyzing product sentiment.ipynb](https://eventing.coursera.org/api/redirectStrict/XW1YO6oBDAfKXs3pYwitB6S8OmSbsjXsKk5XYrMNZTX65UdoEemnchGbZuUwN4urwHH0FF1q0C7rUgCr4YKNNw.Huwy4pxUyWb4IB8zO9ngaw.iArqh7vym0ag1blZqhZaRj17Z3OSQR4Zp-448DLcQeKhY1tJgJanMQiE8Q54WDE3jZcHRbZdruC4Qy_MdZSCxm6XspMG8s0DL1nVHL9hqmkx1pFGnRhpwCTxgQJhbvUQyOvXvzlfd8OJIR-kX9CAA3gNXW-ST1dTs8uH8HZhMZwvAKX16jQcZtAriVU0hy5CM8mW3vKuMAmcQUNHj1TgGOI_KmR5TPXVu1nY1thsLeAla7I1PZe3q5Pr5QJAO-KLSk1DaDKzVl5NGSgglktCBuWoJYCKPff7ZJVS_kPptCykTPSEF1mcgFWr6ntW7c0_rPy3l4LcR7_JbL-ZJCKQobHQVg1eLrrW7XBE_ch99cHcudO1YCm0BG0w_Kp-TEqXLIl2Uua4uTLq-6VXnxwM8UAYpEOdbfAGSRKXOleHo3S6Gjt0DItIe7kjIi_QIdtL7VhAB3-UkANVkJKTw-GcqXMDRCpX9XPUIn_J1_1kDy3lTtyh4bR8QxCYYYiJR1a9)
- Save both of these files in the same directory (where you are calling IPython notebook from) and unzip the data file.

Now you are ready to get started!

#### _**Note: If you would rather use other ML tools...**_

## 

You are welcome to use any ML tool for this course, such as [scikit-learn](https://eventing.coursera.org/api/redirectStrict/Sy3lss2nzwqCTPV1YCcIc_vVuMlpW8mIUPS_WbzTSEBNnKRa_aXtolMLxtyOR4ThjLjjQkjABHzhMlLclen18Q.mHqMnv0rhlSNNRKKZWBKOQ.MC8XuZQoGwCrz8jDaqlYYgBTkS6c27NMgJwnTRbkD-NKLd8zHh7YIKbAD_VfKfNnjZecQhcYYl1cXD5RdOHAaKJhHS8W7kHGIkwAVfWVApmQrfTRmu-s9weDCo3A-6adZYe_gRj8df-cp77X2aIPrbAJdEX79xaBMbblA5ICxiUABrQuu1IveIRi8gaK5_vheQmaIXixYHJgG2bByhgYweEI4j_FW3kJe_qJhPP3cLDOJIHcb2I3W_HOo6wwhGn7oxMpg2hRTLKUxVqM36xeksPPU41VW3DkSX0mniZG2t8). Though, as discussed in the intro module,_ we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)_

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in [open-source under a permissive BSD license](https://eventing.coursera.org/api/redirectStrict/FWvrFKHqSccB0PmlC6vdOGxZX-IgOkGFrIBHNQFv8YL7n16pjIIaBCfXE6N6o3J-GVRIPwXWGd-jokeZ82t-OA.9ZbPB_0u0lyAEtiI-eHidQ.0V-cV7DuFtyUvkTVGml6RQhnduu7Yrvu4W-wsAJIin6GILZbfsTGonXYdQOY7VtBeGso9_qJBaogTnc3vnRSU39kM6DFJ6mWwD4uaERda62Kgp8zn-QT-055c88eWtuprX008Bk2ayLOlKm27H60ec8Ldb_JZXtX4nArPjAdFSlMOle7qzmbbAtr08XkL1XMddq7Be2c8AHkcUTEMHttQSCxMJzzhZPm7udqXpO6FxYJ1xu_Ulj5cDVgv01W2cUqad59EQpu4PpliuYFcBPR6ALzkYjDcGqcvjmE3ymvIu8). So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use [Pandas](https://eventing.coursera.org/api/redirectStrict/L42mCO45hMc2hIHM-3XQ9AbBJfwrnnzOdonEn4sL7VXn-ETKcUERMyN5Wrsu3oo8bhpovrhFQWVWWRhf5cttXw.F9gYl7lkijdr-CNDbFRfsA.TwaXDMp-IVVSvRPM-hM1VM7o851XgAEhO5S_SNGrdlEHQ0J-3OYFofXEWerf3leOTKz2smQ6J1D4sZb7SOrpyHmL7BMak8amhPCC3e77l7LhqvDhCdfjOXAepc3TKt944J5Utd6yAoDZMax8z4UmWJHXb8yiRjDVHen7u1GmSLJ8jOdf2O2y4pNjEMjT4Qmz-CEpXm6sZDvOCL2MkcXmXTHvNfoE1ak5WTnPPEJjNZIMJk7Eo6IqlTbYJ7oNSFXWv7-nvURg_-qr1DtvsnwRYQ) or other options out there: [amazon_baby.csv](https://eventing.coursera.org/api/redirectStrict/Z8o0A7WZd4w5bx0CX4i8QXZ-greiGItqfg8IJZiMaF22wrkeNdKTBlhkdEh8xBMLlZrm0Xa6x6UE6Q9CpbLG-g.-XsJ4jVmAtICohPW7UWd1Q.ucBtoO3vN4xczgxVEhI_B8jm1UU4oHfsUsAwchY_LD7r4PD66QGKpbDPrnm1ksIC-NqliCxroGshf7Zij3ttyrzEokiWo-nW4LeYk906HeNmCdxJQDx6Zv8ONUnBWojKVOuWXp99uLuquvU8It--76v3tXS84JRz3WWwvHz9MxKeBG8mjQPXPfqPxtwMdt08zmjK7cmdCkx8Le8Lfnzf18qlvTFxbNzZkt8zBurCaNoGnULAIeGlI1NIQ3raLSjaxCnkW-XFGJHQdgPw89lB9JRViqCIYpLbzFFkalUV1Z_18Ybn3-hEeqSW-uoxq9WqM5afDAAC8dBcQUAfx7WpI6l9Quv7Hezk_91QnYrkMcoPWau-YHOBHjJvYAar5oR0)

#### Watch the video and explore the IPython notebook on analyzing sentiment

## 

If you haven’t done so yet, before you start, we recommend you watch the video where we go over the IPython notebook on analyzing product sentiment using classifiers from this module. You can then open up the IPython notebook we used and familiarize yourself with the steps we covered in this example.

#### What you will do

## 

Now you are ready! We are going do four tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

In the IPython notebook above, we used the word counts for all words in the reviews to train the sentiment classifier model. Now, we are going to follow a similar path, but only use this subset of the words:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

Often, ML practitioners will throw out words they consider “unimportant” before training their model. This procedure can often be helpful in terms of accuracy. Here, we are going to throw out all words except for the very few above. Using so few words in our model will hurt our accuracy, but help us interpret what our classifier is doing.

1. **Use .apply() to build a new feature with the counts for each of the selected_words:** In the notebook above, we created a column ‘word_count’ with the word counts for each review. Our first task is to create a new column in the products SFrame with the counts for each selected_word above, and, in the process, we will see how the method .apply() can be used to create new columns in our data (our features) and how to use a Python function, which is an extremely useful concept to grasp!

Our first goal is to create a column _products[‘awesome’]_ where each row contains the number of times the word_‘awesome’ _showed up in the review for the corresponding product, and 0 if the review didn’t show up. One way to do this is to look at the each row _‘word_count’_ column and follow this logic:

- If _‘awesome’_ shows up in the word counts for a particular product (row of the products SFrame), then we know how often _‘awesome’_ appeared in the review,

- if _‘awesome’_ doesn’t appear in the word counts, then it didn’t appear in the review, and we should set the count for _‘awesome’_ to 0 in this review.

We could use a for loop to iterate this logic for each row of the products SFrame, but this approach would be really slow, because the SFrame is not optimized for this being accessed with a for loop. Instead, we will use the _.apply() _method to iterate the the logic above for each row of the _products[‘word_count’] _column (which, since it’s a single column, has type SArray). [Read about using the .apply() method on an SArray here.](https://eventing.coursera.org/api/redirectStrict/bjP-uZXwXOkS_ZtukMgO2SQE6naGBxBv0TGhYumSxQDB4Tp7lGnjyd6m1PoyYPdTnGaZ-dLi20ueFz3Q-ADoRw.ZSjzjEsUMtgw6XRMCccwZw.b2EjbQUqDSC1FyOfD0UfBaOUEaNEbHkwN7OTimfpJeatAxtH54zhrV5K4q7iBHp-e6oQBnCLylQGCWRoGzR8xa7JgjRH_hRQv4DddU3hXah2gZXfHlgNNXhailB-8G1ILskOJdTjHevK-J1HoQ4vP2f7_j_CkT9RhGSpQahmpNd8u2VDl6S5eNe9VC9uYFV-1sRbSdlO_QKNJR6_Td96ggPd8vZtJCYC6qfcoVgX711HIdqxks7egrGLk1DiltDT-Kt9JcOjxUa-p5PGi82unLRrEk0uTTZ-OCg0i4hDhnEDF0QiSfYQY8pDzCpvbOQ6PKL0mCULaM-6PhD4WgJzrxQkxaFQBY4SjlJcvhbP5vfcGxRrg1L82JASEhE0mc2LthKnGEze0VB4ERqIapHFsg)

We are now ready to create our new columns:

- First, you will use a Python function to define the logic above. You will write a function called *awesome_count* which takes in the word counts and returns the number of times *‘awesome’* appears in the reviews.

A few tips:

i. Each entry of the ‘word_count’ column is of [Python type dictionary](https://eventing.coursera.org/api/redirectStrict/Lc1CQ3GGoVxSI0RtfnlaMr0ug16cNMwqWyGXPh2VaPvcHYfW3Hn6jo0Yj0o93T3XRtxOYXWDTwefURezQ_DkYg.V18-90iCn63DgCd_sxFXKA.SVaBdeWYKUpswJV9P4fUsQ3KiJIjWrUUnkdmPwha4q4sSwMjunmiab3yfiLHNhak5HFOMgE-y1GgtpDXZOd4te_1HWeMNS8DTdEptGRaRkhMovAHOh49dplrTaxa9IwZSU6BcMpDMXjlYys3Yo0nRKdUIWdtudpEkTPw7Z66Ge7htjha0QbAGPb-Fycy_I8UjiP8sP_VmJLyxy3KhNhaYwInXUalh8P0LQqorL2CvbOp-TPZiLAUQ4pLDLgQqppD_tk1pVlI7xvA-hQMui6bPCcC2_tNii6tmnjdCEwrlJ6sdtV6FUeccyzY745LYERNxaOEflGzxTcYs-cg_ckbL53HfsROv0CTMK_Qpw14XkQ-HCR85HZ6SmXie0LPniAu).

ii. If you have a dictionary called _dict_, you can access a field in the dictionary using:

```
dict[‘awesome’]
```

but only if _‘awesome’_ is one of the fields in the dictionary, otherwise you will get a nasty error.

iii. In Python, to test if a dictionary has a particular field, you can simply write:

```
if ‘awesome’ in dict
```

In our case, if this condition doesn’t hold, the count of _‘awesome’ _should be 0.

Using these tips, you can now write the awesome_count function.

- Next, you will use _.apply()_ to iterate _awesome_count_ for each row of _products[‘word_count’] _and create a new column called _‘awesome’_ with the resulting counts. Here is what that looks like:

```
products[‘awesome’] = products[‘word_count’].apply(awesome_count)
```

And you are done! Check the _products_ SFrame and you should see the new column you just create.

- Repeat this process for the other 11 words in _selected_words_. (Here, we described a simple procedure to obtain the counts for each _selected_word_. There are other more efficient ways of doing this, and we encourage you to explore this further.)
- Using the _.sum() _method on each of the new columns you created, answer the following questions: Out of the _selected_words_, which one is most used in the dataset? Which one is least used? _**Save these results to answer the quiz at the end.**_

2. **Create a new sentiment analysis model using only the selected_words as features:** In the IPython Notebook above, we used word counts for all words as features for our sentiment classifier. Now, you are just going to use the _selected_words_:

- Use the same train/test split as in the IPython Notebook from lecture:

```
train_data,test_data = products.random_split(.8, seed=0)
```

- Train a logistic regression classifier (use _graphlab.logistic_classifier.create_) using just the_ selected_words_. Hint: you can use this parameter in the _.create()_ call to specify the features used to be exactly the new columns you just created:

```
features=selected_words
```

Call your new model: _selected_words_model_.

- You will now examine the weights the learned classifier assigned to each of the 11 words in _selected_words_and gain intuition as to what the ML algorithm did for your data using these features. In GraphLab Create, a learned model, such as the selected_words_model, has a field 'coefficients', which lets you look at the learned coefficients. You can access it by using:

```
selected_words_model[‘coefficients’]
```

The result has a column called ‘value’, which contains the weight learned for each feature.

Using this approach, sort the learned coefficients according to the ‘value’ column using _.sort()_. Out of the 11 words in _selected_words_, which one got the most positive weight? Which one got the most negative weight? Do these values make sense for you? **_Save these results to answer the quiz at the end._**

3. **Comparing the accuracy of different sentiment analysis model: **Using the method

```
.evaluate(test_data)
```

What is the accuracy of the_ selected_words_model_ on the _test_data_? What was the accuracy of the_sentiment_model_ that we learned using all the word counts in the IPython Notebook above from the lectures? What is the accuracy _majority_ class classifier on this task? How do you compare the different learned models with the baseline approach where we are just predicting the majority class? _**Save these results to answer the quiz at the end.**_

_Hint: we discussed the majority class classifier in lecture, which simply predicts that every data point is from the most common class. This is baseline is something we definitely want to beat with models we learn from data._

4. **Interpreting the difference in performance between the models: **To understand why the model with all word counts performs better than the one with only the _selected_words_, we will now examine the reviews for a particular product.

- We will investigate a product named _‘Baby Trend Diaper Champ’_. (This is a trash can for soiled baby diapers, which keeps the smell contained.)
- Just like we did for the reviews for the giraffe toy in the IPython Notebook in the lecture video, before we start our analysis you should select all reviews where the product name is _‘Baby Trend Diaper Champ’_. Let’s call this table _diaper_champ_reviews_.
- Again, just as in the video, use the _sentiment_model _to predict the sentiment of each review in_diaper_champ_reviews_ and sort the results according to their _‘predicted_sentiment’_.
- What is the _‘predicted_sentiment’_ for the most positive review for _‘Baby Trend Diaper Champ’ _according to the sentiment_model from the IPython Notebook from lecture? _**Save this result to answer the quiz at the end.**_
- Now use the _selected_words_model _you learned using just the _selected_words _to predict the sentiment most positive review you found above. _Hint: if you sorted the diaper_champ_reviews in descending order (from most positive to most negative), this command will be helpful to make the prediction you need:_

```
selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')
```

_**Save this result to answer the quiz at the end.**_

- Why is the _predicted_sentiment_ for the most positive review found using the model with all word counts (_sentiment_model_) much more positive than the one using only the _selected_words (selected_words_model)_? _Hint: examine the text of this review, the extracted word counts for all words, and the word counts for each of the selected_words, and you will see what each model used to make its prediction._ _**Save this result to answer the quiz at the end.**_
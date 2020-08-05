import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import streamlit as st
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler,SMOTE
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC,ClassPredictionError
from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import max_error,r2_score,mean_squared_error,mean_absolute_error,explained_variance_score
from yellowbrick.regressor import PredictionError,ResidualsPlot
import warnings
warnings.filterwarnings("ignore")


@st.cache(persist=True,allow_output_mutation=True)
def load_dataset(name):
    data = pd.read_csv(name+".csv",index_col=None)
    return data


@st.cache(persist=True)
def split(df,target,test_size):
    y = df[target]
    x = df.drop(labels=[target],axis=1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=0)
    return x_train,x_test,y_train,y_test


def view_metrics_bin(y_test,pred):
    st.subheader("Metrics Evaluation: ")
    # Writing different metrics in the screen.
    if "Accuracy" in metrics:
        st.write("Accuracy of", model, "on", dataset, " data is: ", accuracy_score(y_test, pred))
    if "F1 Score" in metrics:
        st.write("F1 Score of", model, "on", dataset, " data is: ", f1_score(y_test, pred))
    if "Precision Score" in metrics:
        st.write("Precision Score of", model, "on", dataset, " data is: ", precision_score(y_test, pred))
    if "Recall Score" in metrics:
        st.write("Recall Score of", model, "on", dataset, " data is: ", recall_score(y_test, pred))
    if "ROC-AUC Score" in metrics:
        st.write("ROC-AUC Score of", model, "on", dataset, " data is: ", roc_auc_score(y_test, pred))


def view_metrics_mul(y_test,pred):
    st.subheader("Metrics Evaluation: ")
    # Writing different metrics in the screen.
    if "Accuracy" in metrics:
        st.write("Accuracy of", model, "on", dataset, " data is: ", accuracy_score(y_test, pred))
    if "F1 Score" in metrics:
        st.write("F1 Score of", model, "on", dataset, " data is: ", f1_score(y_test, pred,average="macro"))
    if "Precision Score" in metrics:
        st.write("Precision Score of", model, "on", dataset, " data is: ", precision_score(y_test, pred,average="macro"))
    if "Recall Score" in metrics:
        st.write("Recall Score of", model, "on", dataset, " data is: ", recall_score(y_test, pred,average="macro"))


def view_metrics_reg(y_test,pred):
    st.subheader("Metrics:")
    if "Max Error" in metrics:
        st.write("Max Error of",model,"on",dataset," data is: ",max_error(y_test,pred))
    if "Mean Squared Error" in metrics:
        st.write("Mean Squared Error of", model, "on", dataset, " data is: ", mean_squared_error(y_test, pred))
    if "Mean Absolute Error" in metrics:
        st.write("Mean Absolute Error of", model, "on", dataset, " data is: ", mean_absolute_error(y_test, pred))
    if "R2 Score" in metrics:
        st.write("R2 Score of", model, "on", dataset, " data is: ", r2_score(y_test, pred))
    if "Variance Explained" in metrics:
        st.write("Variance Explained by", model, "on", dataset, " data is: ", explained_variance_score(y_test, pred))


def show_plots_bin(plot,clf,x_test,y_test):
    st.subheader("Plots:")
    if "Confusion Matrix" in plot:
        plot_confusion_matrix(clf, x_test, y_test)
        st.pyplot()
    if "ROC Curve" in plot:
        plot_roc_curve(clf, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in plot:
        plot_precision_recall_curve(clf, x_test, y_test)
        st.pyplot()


def show_plots_mul(plot,clf,x_test,y_test):
    st.subheader("Plots:")
    if "Confusion Matrix" in plot:
        plot_confusion_matrix(clf, x_test, y_test,)
        st.pyplot()
    if "Class Prediction Error" in plot:
        model = ClassPredictionError(clf)
        model.fit(x_train,y_train)
        model.score(x_test, y_test)
        model.show()
        st.pyplot()
    if "ROC Curve" in plot:
        model = ROCAUC(clf)
        model.score(x_test,y_test)
        model.show()
        st.pyplot()


def show_plots_reg(plot,clf,x_test,y_test):
    st.subheader("Plots:")
    if "Prediction Error" in plot:
        model = PredictionError(clf)
        model.score(x_test,y_test)
        model.show()
        st.pyplot()
    if "Residuals" in plot:
        model = ResidualsPlot(clf)
        model.score(x_test, y_test)
        model.show()
        st.pyplot()


@st.cache(persist=True)
def ros(x_train,y_train):
    sm = RandomOverSampler()
    x_sm, y_sm = sm.fit_sample(x_train, y_train)
    return x_sm,y_sm


@st.cache(persist=True)
def smote(x_train,y_train):
    sm = SMOTE()
    x_sm, y_sm = sm.fit_sample(x_train, y_train)
    return x_sm, y_sm


st.title("Machine Learning Workout")
st.write("ML Workout is a platform for machine learning enthusiasts to practice and learn machine learning by practising\
 and observing the effects of training and tuning various machine learning algorithms on simple datasets. Here you will \
  learn how a major part of real machine learning projects is done by working out your way to build a model with the best \
  performance.")
st.write("Select appropriate options from the sidebar.")
st.sidebar.subheader("Choose Options:")
prob_type = st.sidebar.selectbox("Problem Type: ",("Binary Classification","Multi-Class Classification","Regression"))

if prob_type == "Binary Classification":
    st.subheader(prob_type+":")
    st.write("Binary or binomial classification is the task of classifying the elements of a given set into two groups\
     (predicting which group each one belongs to) on the basis of a classification rule.")
    dataset = st.sidebar.selectbox("Sample Datasets:", ('Mushrooms',"Titanic"))
    show_raw_data = st.sidebar.checkbox("Show Raw Data")
    show_pp_data = st.sidebar.checkbox("Show Pre-processed Data")
    test_size = st.sidebar.number_input("Test Set Size:", 0.25, 0.99, step=0.1, key="test_size")
    oversample = st.sidebar.checkbox("Over-sample Minority Class")
    show_data_info = st.sidebar.checkbox("Show Data Info")
    if dataset == "Mushrooms":
        target = "target"
        st.subheader(dataset + " Dataset:")
        st.write("This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled \
        mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, \
        definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the \
        poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a \
        mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.")
        st.write("The task at hand is to classify the Mushrooms into edible and poisonous given various attributes.")
        st.write("More info: This data is approximately balanced and all the features have been Encoded to make training\
                 and evaluation easier.")
        st.subheader("Pre-processing Steps:")
        st.write("* This is a very simple dataset with all categorical variables. Therefore all the features were simply\
         Label Encoded.")
        st.write("(Source: UCI Machine Learning Repository)")
        target_values = ["Edible", "Poisonous"]
        if show_raw_data:
            st.subheader("Raw Dataset:")
            st.write(pd.read_csv("agaricus-lepiota.data").head())
    if dataset == "Titanic":
        target = "Survived"
        st.subheader(dataset + " Dataset:")
        st.write("On April 15, 1912, the largest passenger liner ever made collided with an iceberg during her maiden voyage\
            . When the Titanic sank it killed 1502 out of 2224 passengers and crew. This sensational tragedy shocked the\
             international community and led to better safety regulations for ships. One of the reasons that the shipwreck \
             resulted in such loss of life was that there were not enough lifeboats for the passengers and crew. Although there\
              was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than\
               others.The Titanic Dataset contains data for 887 of the real Titanic passengers. Each row represents one person. ")
        st.write("Our Classification task at hand is to predict whether a individual will survive or not.")
        st.write("More Info: The dataset is imbalanced (Not Survived > Survived). All the null values have been handled and \
            pre-processing has been done.")
        st.subheader("Pre-processing Steps:")
        st.write("* Null values handled.\n * Highly Correlated features removed.\n * All the columns were transformed to\
         categorical for ease in training.\n * The categorical features were then One Hot Encoded or Label Encoded appropriately.")
        st.write("(Source: kaggle.com)")
        target_values = ["Not Survived","Survived"]
        if show_raw_data:
            st.subheader("Raw Dataset:")
            st.write(pd.read_csv("titanic_raw.csv").head())

    # add more elif statements for other datasets.
    data = load_dataset(dataset)
    x_train,x_test,y_train,y_test = split(data,target,test_size)

    if show_pp_data:
        st.subheader("Preprocessed Dataset:")
        st.write(data.head())

    if oversample:
        oversample_algo = st.sidebar.selectbox("Algorithm for Over-sampling:",("Random Over Sampler","SMOTE"),key = "oversample_algo")
        if oversample_algo == "Random Over Sampler":
            x_train,y_train = ros(x_train,y_train)
        if oversample_algo == "SMOTE":
            x_train,y_train = smote(x_train,y_train)

    if show_data_info:
        st.subheader("More Dataset Information:")
        st.write("Shape of Training Data: ",x_train.shape)
        st.write("Shape of Testing Data: ",x_test.shape)
        st.write("Distribution of classes in Training Data:")
        sns.countplot(y_train)
        st.pyplot()
        st.write("Distribution of classes in Test Data:")
        sns.countplot(y_test)
        st.pyplot()

    model = st.sidebar.selectbox("Classifier: ",("Logistic Regression","Random Forest","Support Vector Machine (SVM)","Gaussian Naive Bayes","K Nearest Neighbors"))
    st.subheader(model+":")
    metrics = st.sidebar.multiselect("Metric: (Select one or more)",("Accuracy","F1 Score","Precision Score","Recall Score","ROC-AUC Score"),key="metrics")
    plot = st.sidebar.multiselect("Plots: (Select one or more) ",("Confusion Matrix", "Precision-Recall Curve", "ROC Curve"), key="plot")
    # Add more models.

    if model == "Logistic Regression":
        st.write("Logistic Regression, also known as Logit Regression or Logit Model, is a mathematical model used in\
             statistics to estimate (guess) the probability of an event occurring having been given some previous data. Logistic\
              Regression works with binary data, where either the event happens (1) or the event does not happen (0).")
        st.image("Logistic-curve.png", caption="Logistic Curve")
        C = st.sidebar.number_input("C ",0.01,10.0,key="C")
        max_iter = st.sidebar.number_input("Maximum Number of Iterations:",100,1000,step=10,key="max_iter")
        class_weight_0 = st.sidebar.number_input("Class Weight-- "+target_values[0],1,10,step=1,key="class_weight_0")
        class_weight_1 = st.sidebar.number_input("Class Weight-- "+target_values[1], 1, 10, step=1, key="class_weight_1")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("C = ",C)
            st.write("Maximum number of iterations: ",max_iter)
            st.write("Class Weight: ","{ 0:",class_weight_0,",","1: ",class_weight_1,"}")

            clf = LogisticRegression(C=C,solver='liblinear',max_iter=max_iter,class_weight={0:class_weight_0,1:class_weight_1},n_jobs=-1)
            clf.fit(x_train,y_train)
            pred = clf.predict(x_test)

            view_metrics_bin(y_test,pred)
            show_plots_bin(plot,clf,x_test,y_test)

    if model == "Random Forest":
        st.write("It is an ensemble tree-based learning algorithm. The Random Forest Classifier is a set of decision\
         trees from randomly selected subset of training set. It aggregates the votes from different decision trees to\
          decide the final class of the test object.")
        st.image("random-forest.png")
        n_estimators = st.sidebar.number_input("Number of Trees:",10,1000,step=10,key="n_estimators")
        criterion = st.sidebar.selectbox("Criteria for Splitting:",("gini","entropy"))
        max_depth = st.sidebar.number_input("Maximum depth of trees:", 2, 10, step=1, key="max_depth")
        class_weight_0 = st.sidebar.number_input("Class Weight-- "+target_values[0],1,10,step=1,key="class_weight_0")
        class_weight_1 = st.sidebar.number_input("Class Weight-- "+target_values[1], 1, 10, step=1, key="class_weight_1")
        bootstrap = st.sidebar.checkbox("Bootstrap samples")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Number of Trees (n_estimators):",n_estimators)
            st.write("Criteria for splitting: ",criterion)
            st.write("Maximum depth of trees: ",max_depth)
            st.write("Bootstrap samples: ",bootstrap)
            st.write("Class Weight: ","{ 0:",class_weight_0,",","1: ",class_weight_1,"}")

            clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,bootstrap=bootstrap,class_weight={0:class_weight_0,1:class_weight_1},n_jobs=-1)
            clf.fit(x_train,y_train)
            pred = clf.predict(x_test)

            view_metrics_bin(y_test,pred)
            show_plots_bin(plot, clf, x_test, y_test)

    if model == "Support Vector Machine (SVM)":
        st.write("In machine learning, support-vector machines (SVMs, also known as support-vector networks) are supervised \
        learning models with associated learning algorithms that analyze data used for classification and regression \
        analysis. In addition to performing linear classification, SVMs can efficiently perform a non-linear \
        classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional \
        feature spaces.")
        st.image("svm.png", caption="Kernel Machine")
        st.warning("The time complexity of Support Vector Machines is O(n\u00b3). Therefore it may take a few seconds/ minutes\
                   to run. Please be patient.")
        C = st.sidebar.number_input("C ", 0.01, 10.0, key="C")
        kernel = st.sidebar.radio("Select Kernel:", ("linear", "poly", "rbf", "sigmoid"), key="kernel")
        class_weight_0 = st.sidebar.number_input("Class Weight-- " + target_values[0], 1, 10, step=1,key="class_weight_0")
        class_weight_1 = st.sidebar.number_input("Class Weight-- " + target_values[1], 1, 10, step=1,key="class_weight_1")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("C = ", C)
            st.write("Kernel: ", kernel)
            st.write("Class Weight: ", "{ 0:", class_weight_0, ",", "1: ", class_weight_1, "}")

            clf = SVC(C=C, kernel=kernel, class_weight={0: class_weight_0, 1: class_weight_1})
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_bin(y_test,pred)
            show_plots_bin(plot, clf, x_test, y_test)

    if model == "Gaussian Naive Bayes":
        st.write("Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to \
        problem instances, represented as vectors of feature values, where the class labels are drawn from some finite \
        set. All naive Bayes classifiers assume that the value of a particular feature is independent of the value of \
        any other feature, given the class variable.")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Gaussian Naive Bayes does not have any hyperpaprameters to tune.")

            clf = GaussianNB()
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_bin(y_test,pred)
            show_plots_bin(plot, clf, x_test, y_test)

    if model == "K Nearest Neighbors":
        st.write("The k-nearest neighbors algorithm (k-NN) is a non-parametric method proposed by Thomas Cover used for \
        classification and regression. In both cases, the input consists of the k closest training examples in the feature space.")
        n_neighbors = st.sidebar.number_input("Number of neighbors to use:",3,200,step=1,key="n_neighbors")
        weights = st.sidebar.radio("Weights:",("uniform","distance"),key="weights")
        algo = st.sidebar.radio("Algorithm:",("auto","ball_tree","kd_tree","brute"))
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Number of nearest neighbors to use: ",n_neighbors)
            st.write("Weights: ",weights)
            st.write("Algorithm to be used for Classification: ",algo)

            clf = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algo,n_jobs=-1)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_bin(y_test,pred)
            show_plots_bin(plot, clf, x_test, y_test)

if prob_type == "Multi-Class Classification":
    st.subheader(prob_type + ":")
    st.write("Multi-class classification or multinomial classification is a classification task with more than two classes; e.g., classify a set of images of fruits which may be\
     oranges, apples, or pears. Multi-class classification makes the assumption that each sample is assigned to one and\
      only one label: a fruit can be either an apple or a pear but not both at the same time.")
    dataset = st.sidebar.selectbox("Sample Datasets:", ('Iris',"Wine Quality"))     # Add more datasets
    show_raw_data = st.sidebar.checkbox("Show Raw Data")
    show_pp_data = st.sidebar.checkbox("Show Pre-processed Data")
    test_size = st.sidebar.number_input("Test Set Size:", 0.25, 0.99, step=0.1, key="test_size")
    oversample = st.sidebar.checkbox("Over-sample Minority Class")
    show_data_info = st.sidebar.checkbox("Show Data Info")
    # Add conditions for dataset.
    if dataset == "Iris":
        target = "target"
        st.subheader(dataset + " Dataset:")
        st.write("The data set contains 3 classes of 50 instances each,where each class refers to a type of iris plant.\
         One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.")
        st.write("The task at hand is to classify the flowers into 3 categories namely Setosa, Virginica and Versicolor\
                 given Sepal Width, Sepal Length, Petal Width, Petal Length.")
        st.write("More info: This data is balanced and all the features have been standardized to make training\
                     and evaluation easier.")
        st.subheader("Pre-processing Steps:")
        st.write("* Target column was Label Encoded.\n * All the features were numerical so they were standardized. ")
        st.write("(Source: UCI Machine Learning Repository)")
        target_values = ["Iris-Setosa", "Iris Virginica","Iris-Versicolor"]
        if show_raw_data:
            st.subheader("Raw Dataset:")
            st.write(pd.read_csv("iris.data").head())
    if dataset == "Wine Quality":
        target = "target"
        st.subheader(dataset + " Dataset:")
        st.write("Wine Quality Dataset has been made publicly available for research consists of individual samples of \
         red and white wine labelled by their quality with 1 being the worst and 8 being the best. The samples are \
         labelled based on various parameters namely: fixed acidity, volatile acidity, citric acid, residual sugar,\
          chlorides, free sulfur dioxide, total sulfur dioxide, density,pH, sulphates, alcohol. For ease of training \
         and evaluation the target variable as been changed from 8 classes to 3 classes (Low:0, Medium:1, High:2). The \
          data has been pre-processed and standardized and highly correlated features have been removed.")
        st.write("More Info: The Dataset is imbalanced and the classes need to be weighted or oversampled for\
                 better predictions.")
        st.subheader("Pre-processing Steps:")
        st.write("* Highly correlated columns were removed.\n * The target column was changed from 8 classes to 3 classes\
          to increase the number of data points per class.\n * The target column was Label Encoded.\n * The numerical \
          features were standardized.")
        st.write("(Source: UCI Machine Learning Repository)")
        target_values = ["Low", "Medium","High"]
        if show_raw_data:
            st.subheader("Raw Dataset:")
            st.write(pd.read_csv("Wine_raw.csv").head())

    data = load_dataset(dataset)
    x_train, x_test, y_train, y_test = split(data, target,test_size)

    if show_pp_data:
        st.subheader("Preprocessed Dataset:")
        st.write(data.head())

    if oversample:
        oversample_algo = st.sidebar.selectbox("Algorithm for Over-sampling:",("Random Over Sampler","SMOTE"),key = "oversample_algo")
        if oversample_algo == "Random Over Sampler":
            x_train,y_train = ros(x_train,y_train)
        if oversample_algo == "SMOTE":
            x_train,y_train = smote(x_train,y_train)

    if show_data_info:
        st.subheader("More Dataset Information:")
        st.write("Shape of Training Data: ",x_train.shape)
        st.write("Shape of Testing Data: ",x_test.shape)
        st.write("Distribution of classes in Training Data:")
        sns.countplot(y_train)
        st.pyplot()
        st.write("Distribution of classes in Test Data:")
        sns.countplot(y_test)
        st.pyplot()

    model = st.sidebar.selectbox("Classifier: ", ("Logistic Regression", "Random Forest", "Support Vector Machine (SVM)", "Gaussian Naive Bayes","K Nearest Neighbors"))
    st.subheader(model + ":")
    metrics = st.sidebar.multiselect("Metric: (Select one or more)",
                                     ("Accuracy", "F1 Score", "Precision Score", "Recall Score"))
    plot = st.sidebar.multiselect("Plots: (Select one or more) ",
                                  ("Confusion Matrix", "Class Prediction Error", "ROC Curve"), key="plot")
    # Add more models.
    if model == "Logistic Regression":
        st.write("Logistic Regression, also known as Logit Regression or Logit Model, is a mathematical model used in\
             statistics to estimate (guess) the probability of an event occurring having been given some previous data. Logistic\
              Regression works with binary data, where either the event happens (1) or the event does not happen (0).")
        st.image("Logistic-curve.png", caption="Logistic Curve")
        C = st.sidebar.number_input("C ",0.01,10.0,key="C")
        max_iter = st.sidebar.number_input("Maximum Number of Iterations:",100,1000,step=10,key="max_iter")
        class_weight_0 = st.sidebar.number_input("Class Weight-- "+target_values[0],1,10,step=1,key="class_weight_0")
        class_weight_1 = st.sidebar.number_input("Class Weight-- "+target_values[1], 1, 10, step=1, key="class_weight_1")
        class_weight_2 = st.sidebar.number_input("Class Weight-- " + target_values[2], 1, 10, step=1,key="class_weight_2")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("C = ",C)
            st.write("Maximum number of iterations: ",max_iter)
            st.write("Class Weight: ","{ 0:",class_weight_0,",","1: ",class_weight_1,",","2: ",class_weight_2,"}")

            clf = LogisticRegression(C=C,max_iter=max_iter,multi_class="ovr",class_weight={0:class_weight_0,1:class_weight_1,2:class_weight_2},n_jobs=-1)
            clf.fit(x_train,y_train)
            pred = clf.predict(x_test)

            view_metrics_mul(y_test,pred)
            show_plots_mul(plot,clf,x_test,y_test)

    if model == "Random Forest":
        st.write("It is an ensemble tree-based learning algorithm. The Random Forest Classifier is a set of decision\
         trees from randomly selected subset of training set. It aggregates the votes from different decision trees to\
          decide the final class of the test object.")
        st.image("random-forest.png")
        n_estimators = st.sidebar.number_input("Number of Trees:",10,1000,step=10,key="n_estimators")
        criterion = st.sidebar.selectbox("Criteria for Splitting:",("gini","entropy"))
        max_depth = st.sidebar.number_input("Maximum depth of trees:", 2, 10, step=1, key="max_depth")
        class_weight_0 = st.sidebar.number_input("Class Weight-- "+target_values[0],1,10,step=1,key="class_weight_0")
        class_weight_1 = st.sidebar.number_input("Class Weight-- "+target_values[1], 1, 10, step=1, key="class_weight_1")
        class_weight_2 = st.sidebar.number_input("Class Weight-- " + target_values[2], 1, 10, step=1,key="class_weight_2")
        bootstrap = st.sidebar.checkbox("Bootstrap samples")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Number of Trees (n_estimators):",n_estimators)
            st.write("Criteria for splitting: ",criterion)
            st.write("Maximum depth of trees: ",max_depth)
            st.write("Class Weight: ", "{ 0:", class_weight_0, ",", "1: ", class_weight_1, ",", "2: ", class_weight_2,"}")
            st.write("Bootstrap samples: ",bootstrap)

            clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,bootstrap=bootstrap,class_weight={0:class_weight_0,1:class_weight_1,2:class_weight_2},n_jobs=-1)
            clf.fit(x_train,y_train)
            pred = clf.predict(x_test)

            view_metrics_mul(y_test,pred)
            show_plots_mul(plot, clf, x_test, y_test)

    if model == "Support Vector Machine (SVM)":
        st.write("In machine learning, support-vector machines (SVMs, also known as support-vector networks) are supervised \
        learning models with associated learning algorithms that analyze data used for classification and regression \
        analysis. In addition to performing linear classification, SVMs can efficiently perform a non-linear \
        classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional \
        feature spaces.")
        st.image("svm.png", caption="Kernel Machine")
        st.warning("The time complexity of Support Vector Machines is O(n\u00b3). Therefore it may take a few seconds \ minutes\
                   to run. Please be patient.")
        C = st.sidebar.number_input("C ", 0.01, 10.0, key="C")
        kernel = st.sidebar.radio("Select Kernel:", ("linear", "poly", "rbf", "sigmoid"), key="kernel")
        class_weight_0 = st.sidebar.number_input("Class Weight-- " + target_values[0], 1, 10, step=1,key="class_weight_0")
        class_weight_1 = st.sidebar.number_input("Class Weight-- " + target_values[1], 1, 10, step=1,key="class_weight_1")
        class_weight_2 = st.sidebar.number_input("Class Weight-- " + target_values[2], 1, 10, step=1,key="class_weight_2")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("C = ", C)
            st.write("Kernel: ", kernel)
            st.write("Class Weight: ", "{ 0:", class_weight_0, ",", "1: ", class_weight_1, ",", "2: ", class_weight_2,"}")

            clf = SVC(C=C, kernel=kernel,class_weight={0:class_weight_0,1:class_weight_1,2:class_weight_2})
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_mul(y_test,pred)
            show_plots_mul(plot, clf, x_test, y_test)

    if model == "Gaussian Naive Bayes":
        st.write("Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to \
        problem instances, represented as vectors of feature values, where the class labels are drawn from some finite \
        set. All naive Bayes classifiers assume that the value of a particular feature is independent of the value of \
        any other feature, given the class variable.")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Gaussian Naive Bayes does not have any hyperparameters to tune.")

            clf = GaussianNB()
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_mul(y_test,pred)
            show_plots_mul(plot, clf, x_test, y_test)

    if model == "K Nearest Neighbors":
        st.write("The k-nearest neighbors algorithm (k-NN) is a non-parametric method proposed by Thomas Cover used for \
        classification and regression. In both cases, the input consists of the k closest training examples in the feature space.")
        n_neighbors = st.sidebar.number_input("Number of neighbors to use:",3,200,step=1,key="n_neighbors")
        weights = st.sidebar.radio("Weights:",("uniform","distance"),key="weights")
        algo = st.sidebar.radio("Algorithm:",("auto","ball_tree","kd_tree","brute"))
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Number of nearest neighbors to use: ",n_neighbors)
            st.write("Weights: ",weights)
            st.write("Algorithm to be used for Classification: ",algo)

            clf = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algo,n_jobs=-1)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_mul(y_test,pred)
            show_plots_mul(plot, clf, x_test, y_test)

if prob_type == "Regression":
    st.subheader(prob_type+":")
    st.write("In statistical modeling, regression analysis is a set of statistical processes for estimating the \
    relationships between a dependent variable (often called the 'outcome variable') and one or more independent \
    variables (often called 'predictors', 'covariates', or 'features'). The most common form of regression analysis is \
    linear regression.\n Regression analysis is primarily used for two conceptually distinct purposes. First, \
    regression analysis is widely used for prediction and forecasting, where its use has substantial overlap with the \
    field of machine learning. Second, in some situations regression analysis can be used to infer causal relationships\
     between the independent and dependent variables.")
    dataset = st.sidebar.selectbox("Sample Datasets:", ('Boston Housing',"Advertising Data"))
    show_raw_data = st.sidebar.checkbox("Show Raw Data")
    show_pp_data = st.sidebar.checkbox("Show Pre-processed Data")
    test_size = st.sidebar.number_input("Test Set Size:", 0.25, 0.99, step=0.1, key="test_size")
    show_data_info = st.sidebar.checkbox("Show Data Info")
    if dataset == "Boston Housing":
        target = "MEDV"
        st.subheader(dataset + " Dataset:")
        st.write("The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data \
        about 14 features for homes from various suburbs in Boston, Massachusetts.For the purposes of this analysis, the \
        following pre-processing steps have been made to the dataset.")
        st.write("The task at hand is to predict the best value of the house given the features.")
        st.subheader("Pre-processing Steps:")
        st.write("* 16 data points have an 'MEDV' value of 50.0. These data points likely contain missing or censored\
         values and have been removed.\n * 1 data point has an 'RM' value of 8.78. This data point can be considered an \
         outlier and has been removed.\n * The features 'RM', 'LSTAT', 'PTRATIO', and 'MEDV' are essential. The\
          remaining non-relevant features have been excluded.\n * The feature 'MEDV' has been multiplicatively scaled \
          to account for 35 years of market inflation.\n * The features have been scaled.")

        st.write("(Source: UCI Machine Learning Repository)")
        if show_raw_data:
            st.subheader("Raw Dataset:")
            st.write(pd.read_csv("boston-raw.csv").head())
    if dataset == "Advertising Data":
        target = "Sales"
        st.subheader(dataset + " Dataset:")
        st.write("This dataset consists of 4 real-valued columns namely: TV, Radio, Newspaper and Sales. The values of\
         the columns denote the amount of money spent in a particular form of advertising.")
        st.write("Our Regression task at hand is to predict the sales given the advertising data.")
        st.write("(Source: kaggle.com)")
        if show_raw_data:
            st.subheader("Raw Dataset:")
            st.write(pd.read_csv("Advertising Data.csv").head())

    data = load_dataset(dataset)
    x_train,x_test,y_train,y_test = split(data,target,test_size)

    if show_pp_data:
        st.subheader("Preprocessed Dataset:")
        st.write(data.head())

    if show_data_info:
        st.subheader("More Dataset Information:")
        st.write("Shape of Training Data: ",x_train.shape)
        st.write("Shape of Testing Data: ",x_test.shape)
        st.write("Distribution of classes in Training Data:")
        sns.distplot(y_train,bins=30)
        st.pyplot()
        st.write("Distribution of classes in Test Data:")
        sns.distplot(y_test,bins=30)
        st.pyplot()

    model = st.sidebar.selectbox("Regressor: ",("Linear Regression","Random Forest","Support Vector Machine (SVM)","K Nearest Neighbors"))
    st.subheader(model+":")
    metrics = st.sidebar.multiselect("Metric: (Select one or more)",("Max Error","Mean Squared Error","Mean Absolute Error","R2 Score","Variance Explained"),key="metrics")
    plot = st.sidebar.multiselect("Plots: (Select one or more) ",("Prediction Error","Residuals"), key="plot")

    if model == "Linear Regression":
        st.write("In statistics, linear regression is a linear approach to modeling the relationship between a scalar\
         response (or dependent variable) and one or more explanatory variables (or independent variables). The case of\
          one explanatory variable is called simple linear regression.")
        st.image("Linear-regression.png", caption="Linear Regression")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Linear Regression is one of the simplest statistical models so it does not have any \
            hyperparameters.")

            clf = LinearRegression(n_jobs=-1)
            clf.fit(x_train,y_train)
            pred = clf.predict(x_test)

            view_metrics_reg(y_test,pred)
            show_plots_reg(plot,clf,x_test,y_test)

    if model == "Random Forest":
        st.write("It is an ensemble tree-based learning algorithm. The Random Forest Classifier is a set of decision\
         trees from randomly selected subset of training set. It aggregates the votes from different decision trees to\
          decide the final class of the test object.")
        st.image("random-forest.png")
        n_estimators = st.sidebar.number_input("Number of Trees:",10,1000,step=10,key="n_estimators")
        criterion = st.sidebar.selectbox("Criteria for Splitting:",("MSE","MAE"))
        max_depth = st.sidebar.number_input("Maximum depth of trees:", 2, 10, step=1, key="max_depth")
        bootstrap = st.sidebar.checkbox("Bootstrap samples")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Number of Trees (n_estimators):",n_estimators)
            st.write("Criteria for splitting: ",criterion)
            st.write("Maximum depth of trees: ",max_depth)
            st.write("Bootstrap samples: ",bootstrap)

            clf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion.lower(),bootstrap=bootstrap,n_jobs=-1)
            clf.fit(x_train,y_train)
            pred = clf.predict(x_test)

            view_metrics_reg(y_test,pred)
            show_plots_reg(plot, clf, x_test, y_test)

    if model == "Support Vector Machine (SVM)":
        st.write("In machine learning, support-vector machines (SVMs, also known as support-vector networks) are supervised \
        learning models with associated learning algorithms that analyze data used for classification and regression \
        analysis. In addition to performing linear classification, SVMs can efficiently perform a non-linear \
        classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional \
        feature spaces.")
        st.image("svm.png", caption="Kernel Machine")
        st.warning("The time complexity of Support Vector Machines is O(n\u00b3). Therefore it may take a few seconds/ minutes\
                   to run. Please be patient.")
        C = st.sidebar.number_input("C ", 0.01, 10.0, key="C")
        kernel = st.sidebar.radio("Select Kernel:", ("linear", "poly", "rbf", "sigmoid"), key="kernel")
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("C = ", C)
            st.write("Kernel: ", kernel)

            clf = SVR(C=C, kernel=kernel,)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_reg(y_test,pred)
            show_plots_reg(plot, clf, x_test, y_test)

    if model == "K Nearest Neighbors":
        st.write("The k-nearest neighbors algorithm (k-NN) is a non-parametric method proposed by Thomas Cover used for \
        classification and regression. In both cases, the input consists of the k closest training examples in the feature space.")
        n_neighbors = st.sidebar.number_input("Number of neighbors to use:",3,200,step=1,key="n_neighbors")
        weights = st.sidebar.radio("Weights:",("uniform","distance"),key="weights")
        algo = st.sidebar.radio("Algorithm:",("auto","ball_tree","kd_tree","brute"))
        run_model = st.sidebar.button("Train and Evaluate")
        if run_model:
            st.subheader("Hyperparameters:")
            st.write("Number of nearest neighbors to use: ",n_neighbors)
            st.write("Weights: ",weights)
            st.write("Algorithm to be used for Classification: ",algo)

            clf = KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algo,n_jobs=-1)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            view_metrics_reg(y_test,pred)
            show_plots_reg(plot, clf, x_test, y_test)
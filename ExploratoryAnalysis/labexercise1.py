import pandas as pd
from scipy import stats
from numpy import cov
from matplotlib import pyplot as plot

print("====== 1. Read CSV file ======")

collist = ["student_id","english","tamil","maths","science"]
# 1. Read File
df = pd.read_csv("E:\Class Notes\ML\Lab\labex1\marks.csv", usecols=collist)
print(df)

print("--- 2. Exploratory Analysis ---")

# 2. Mean, Median, Variance, SD, GM, HM

print("====== A.  Mean, Median, Variance, SD, GM, HM ======")

print("Describe")
print(df.describe())

print("--- Mean ---")
engmean = df["english"].mean()
tammean = df["tamil"].mean()
mathsmean = df["maths"].mean()
scimean = df["science"].mean()

print("English Mean :",engmean , "\nTamil Mean:",tammean, "\nMaths Mean",mathsmean, "\nScience Mean:",scimean)
print("\n")

print("--- Median ---")
engmedian = df["english"].median()
tammedian = df["tamil"].median()
mathsmedian = df["maths"].median()
scimedian = df["science"].median()

print("English Median :",engmedian , "\nTamil Median:",tammedian, "\nMaths Median",mathsmedian, "\nScience Median:",scimedian)
print("\n")

print("--- Variance ---")
engvar = df["english"].var()
tamvar = df["tamil"].var()
mathsvar = df["maths"].var()
scivar = df["science"].var()

print("English Variance :",engvar , "\nTamil Variance:",tamvar, "\nMaths Variance",mathsvar, "\nScience Variance:",scivar)
print("\n")

print("--- Standard Deviation ---")
engsd = df["english"].std()
tamsd = df["tamil"].std()
mathssd = df["maths"].std()
scisd = df["science"].std()

print("English SD :",engsd , "\nTamil SD:",tamsd, "\nMaths SD",mathssd, "\nScience SD:",scisd)
print("\n")

print("--- Geometric Mean ---")

print("English Array: \n",df["english"].to_numpy())

enggm = stats.gmean(df["english"].to_numpy())
tamgm = stats.gmean(df["tamil"].to_numpy())
mathsgm = stats.gmean(df["maths"].to_numpy())
scigm = stats.gmean(df["science"].to_numpy())

print("English GM :",enggm , "\nTamil GM:",tamgm, "\nMaths GM",mathsgm, "\nScience GM:",scigm)
print("\n")

print("--- Harmonic Mean ---")

enghm = stats.hmean(df["english"].to_numpy())
tamhm = stats.hmean(df["tamil"].to_numpy())
mathshm = stats.hmean(df["maths"].to_numpy())
scihm = stats.hmean(df["science"].to_numpy())

print("English HM :",enghm , "\nTamil HM:",tamhm, "\nMaths HM",mathshm, "\nScience HM:",scihm)
print("\n")

# 3. Skewness and Kurtosis
print("====== B. Skewness and Kurtosis ======")
print("--- Skewness ---")

engskew = stats.skew(df["english"].to_numpy())
tamkew = stats.skew(df["tamil"].to_numpy())
mathskew = stats.skew(df["maths"].to_numpy())
scikew = stats.skew(df["science"].to_numpy())

print("English Skewness :",engskew , "\nTamil Skewness:",tamkew, "\nMaths Skewness",mathskew, "\nScience Skewness:",scikew)
print("\n")

print("--- Kurtosis ---")

engskurt = stats.kurtosis(df["english"].to_numpy())
tamkurt = stats.kurtosis(df["tamil"].to_numpy())
mathkurt = stats.kurtosis(df["maths"].to_numpy())
scikurt = stats.kurtosis(df["science"].to_numpy())

print("English Kurtosis :",engskurt , "\nTamil Kurtosis:",tamkurt, "\nMaths Kurtosis",mathkurt, "\nScience Kurtosis:",scikurt)
print("\n")

# 3. Correlation and Covariance between English and Tamil Marks
print("====== C. Correlation and Covariance between English and Tamil Marks ======")
engmarks = df["english"].to_numpy()
tamilmarks = df["tamil"].to_numpy()
pcorr,pval = stats.pearsonr(engmarks,tamilmarks)
print("Pearson Correlation of English and Tamil Marks :", pcorr )
scorr,spval = stats.spearmanr(engmarks,tamilmarks)
print("Spearman Correlation of English and Tamil Marks :", scorr)
print("Covariance Matrix: \n", cov(engmarks, tamilmarks))
print("\n")

# 4. Correlation and Covariance between English and Science Marks
print("====== D. Correlation and Covariance between English and Science Marks ======")
engmarks = df["english"].to_numpy()
scilmarks = df["science"].to_numpy()
pcorr,pval = stats.pearsonr(engmarks,scilmarks)
print("Pearson Correlation of English and Science Marks :", pcorr )
scorr,spval = stats.pearsonr(engmarks,scilmarks)
print("Spearman Correlation of English and Science Marks :", scorr)
print("Covariance Matrix : \n", cov(engmarks, scilmarks))

print("\n")

collist1 = ["english","tamil","maths","science"]
# 1. Read File
df1 = pd.read_csv("E:\Class Notes\ML\Lab\labex1\marks.csv", usecols=collist1)
print("--- Complete Correlation Matrix ---")
print(df1.corr())
print("\n")


# E. Do the Bar Plot, Pie Chart, Histogram, Box Plot

#Bar Plot

print("====== E. Do the Bar Plot, Pie Chart, Histogram, Box Plot ======")

plotdata = pd.DataFrame({
    "english" :df["english"].to_numpy(),
    "tamil" :df["tamil"].to_numpy(),
    "maths" :df["maths"].to_numpy(),
    "science" :df["science"].to_numpy()},
    index = df["student_id"].to_numpy())
    
plotdata.plot(kind="bar",figsize=(15, 8))
plot.title("Bar Plot for Students Marks")
plot.xlabel("Student Id")
plot.ylabel("Marks")
plot.show()

# Pie Chart

collist = ["english","tamil","maths","science"]
df = pd.read_csv("E:\Class Notes\ML\Lab\labex1\marks.csv", usecols=collist)
for ind in df.index:
    fig, ax = plot.subplots(1,1)
    fig.set_size_inches(5,5)
    df.iloc[ind].plot(kind='pie', ax=ax, autopct='%1.1f%%', title ="Pie Char for Student :"+str(ind+1))
    ax.set_ylabel('')
    ax.set_xlabel('')
    plot.show()


# Histogram

enghist = df['english'].hist()
plot.title('Histogram for english')
plot.xlabel('english')
plot.ylabel('Frequency of english')
plot.show()

enghist = df['tamil'].hist()
plot.title('Histogram for tamil')
plot.xlabel('tamil')
plot.ylabel('Frequency of tamil')
plot.show()

enghist = df['maths'].hist()
plot.title('Histogram for Maths')
plot.xlabel('maths')
plot.ylabel('Frequency of Maths')
plot.show()

enghist = df['science'].hist()
plot.title('Histogram for science')
plot.xlabel('science')
plot.ylabel('Frequency of science')
plot.show()

# Box Plot
df.plot.box("Box Plot", grid = False, title = "Box Plot for Student Marks")
plot.show()


print("====== F. Do the Scatter Diagram between English and Science Marks======")

# F. Do the Scatter Diagram between English and Science Marks
engmarks = df["english"].values.tolist()
scilmarks = df["science"].to_numpy()
plot.scatter(engmarks,scilmarks, alpha=0.5, c="blue")
plot.title("Scatter Diagram between English and Science Marks")
plot.xlabel("English")
plot.ylabel("Science")
plot.show()

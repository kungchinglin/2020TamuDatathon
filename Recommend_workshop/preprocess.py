import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r'C:\Users\10330\Downloads\applicants-2.csv')

target_name = 'workshop'
feature_names = ['num_hackathons_attended', 'majors-aerospace', 'majors-agricultural', 'majors-analytics', 'majors-applied',
                 'majors-architecture', 'majors-astronomy', 'majors-biochemistry', 'majors-biological', 'majors-biophysics',
                 'majors-business', 'majors-chemical', 'majors-civil', 'majors-computer', 'majors-crop',
                 'majors-data', 'majors-distribution', 'majors-electrical', 'majors-electronic', 'majors-engineering',
                 'majors-entomology', 'majors-finance', 'majors-genetics', 'majors-industrial', 'majors-information',
                 'majors-landscape', 'majors-management', 'majors-mathematics', 'majors-mechanical', 'majors-microbiology',
                 'majors-neuroscience', 'majors-pathology', 'majors-petroleum', 'majors-physics', 'majors-planning',
                 'majors-plant', 'majors-science', 'majors-sciences', 'majors-soil', 'majors-statistics', 'majors-systems',
                 'majors-technology', 'majors-urban', 'relavent_industries-aerospace', 'relavent_industries-consulting',
                 'relavent_industries-education', 'relavent_industries-energy', 'relavent_industries-finance',
                 'relavent_industries-healthcare', 'relavent_industries-insurance', 'relavent_industries-public_policy',
                 'relavent_industries-retail', 'relavent_industries-sports', 'relavent_industries-technology',
                 'relavent_industries-transportation', 'technology_experience-cloud', 'technology_experience-dev_ops',
                 'technology_experience-excel', 'technology_experience-full_stack', 'technology_experience-keras',
                 'technology_experience-learn', 'technology_experience-matlab', 'technology_experience-numpy',
                 'technology_experience-pandas', 'technology_experience-python', 'technology_experience-pytorch',
                 'technology_experience-scikit', 'technology_experience-sql', 'technology_experience-tableau',
                 'technology_experience-tensorflow', 'class-freshman', 'class-junior', 'class-master', 'class-phd',
                 'class-postdoc', 'class-senior', 'class-sophomore']
target = df[target_name]
features = df[feature_names]
print(df)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3, random_state=0)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_std, Y_train)
print(lr.predict(X_test_std))
# print(lr.predict_proba(X_test_std))

for predict in lr.predict_proba(X_test_std):
    gapBest = max(predict)
    print(gapBest)



# forest = RandomForestClassifier(criterion='entropy', n_estimators=7, random_state=3, n_jobs=2)
# forest.fit(X_train, Y_train)
# print(forest.score(X_test, Y_test))


# tree = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=0)
# tree.fit(X_train, Y_train)
# print(tree.score(X_test, Y_test))

# clf = svm.SVC(kernel='poly', C=2, gamma='auto')
# clf.fit(X_train, Y_train)
# print(clf.score(X_test, Y_test))

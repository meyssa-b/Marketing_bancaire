#Import Packages
library(dplyr)
library(ggplot2)
library(ggthemes)
library(broom)
library(ROCR)
library(caret)
library(MASS)
library(randomForest)


chemin_dossier = "C:/Users/meyss/Documents"
setwd(chemin_dossier)
df=read.csv("bank-full.csv",sep=";")

#Transformation en facteur des variables catégorielles
var=c("job","marital","education","default","housing","loan","contact","month","poutcome","y")
for(i in var){
  df[,i]=as.factor(df[,i])
}
######################Liaisons entr les prédicteurs et la variable d'intérêt######################
summary(df)

#Répartition de la variable d'interet
prop.table(table(df$y))*100
table(df$y)

#Croisement variable default/y
prop.table(table(df$default,df$y),margin=1)*100
table(df$default,df$y)

#Croisement variable balance/y
quantile(df$balance,0.1*1:10)
df_clean_v1=df[which(df$balance<7500&df$balance>-4500),]
ggplot(df_clean_v1,aes(y,balance))+geom_boxplot()+theme_economist()

#Croisement variable housing/y
prop.table(table(df$housing,df$y),margin=1)*100
table(df$housing,df$y)

#Croisement variable month/y
round(prop.table(table(df$month,df$y),margin = 1)*100,1)

#Croisement variable duration/y
df_clean=df[which(df$duration<4000),]
ggplot(df_clean,aes(y,duration))+geom_boxplot()+theme_economist()

#Croisement variable poutcome/y
prop.table(table(df$poutcome,df$y),margin = 1)*100

#Croisement variable contact/y
prop.table(table(df$contact,df$y),margin = 1)*100

#Création de la base final
df_vf=df[which(df$balance<7500&df$balance>-4500&df$duration<4000),]



######################Création des échantillons de test et d'entraînement######################


df_no=df_vf[which(df_vf$y=="no"),]
df_yes=df_vf[which(df_vf$y=="yes"),]

nb_app=0.7*nrow(df_yes)
nb_test=round(0.3*nrow(df_yes),0)

split_no<- sample(1:nrow(df_no), nb_app)
split_yes<- sample(1:nrow(df_yes), nb_app)
split_no_t<- sample(1:nrow(df_no), nb_test)

df_train=rbind(df_no[split_no,],df_yes[split_yes,])
df_test=rbind(df_no[split_no_t,],df_yes[-split_yes,])

table(df_train$y)
table(df_test$y)


######################Prédiction######################
######################LOGIT


#Création des 3 modèles avec un modèle fait à partir d'une sélection pas à pas
model_glm1<-glm(y~balance+duration+month+poutcome+campaign+job+contact+default+housing, family = "binomial", data=df_train)
model_glm2<-glm(y~., family = "binomial", data=df_train)

#m0=glm(y~1,data=df_train,family = "binomial")
#mf=glm(y~.,data=df_train,family = "binomial")
#mboth=step(m0, scope = list(upper=mf),data=df_train,direction="both",trace = F)
#summary(mboth)

model_glm3<-glm(y~duration + poutcome + month + contact + housing + 
                  job + campaign + loan + marital + balance + education + day + 
                  previous, family = "binomial", data=df_train)


#Analyse des résultats sur la qualité d'ajustement des modèles
library(pscl)
list(model_glm1=pscl::pR2(model_glm1)["McFadden"],
     model_glm2=pscl::pR2(model_glm2)["McFadden"],
     model_glm3=pscl::pR2(model_glm3)["McFadden"])
#deviance, AIC,BIC
list(model_glm1=broom::glance(model_glm1),
     model_glm2=broom::glance(model_glm2),
     model_glm3=broom::glance(model_glm3))


#Création des modèles avec une cross-validation
cv_model1 <- train(
  y~balance+duration+month+poutcome+campaign+job+contact+default+housing, 
  data = df_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)



cv_model2 <- train(
  y~., 
  data = df_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)



cv_model3 <- train(
  y~duration + poutcome + month + contact + housing + 
    job + campaign + loan + marital + balance + education + day + 
    previous, 
  data = df_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)


#Sortie de la répartition de la précision de chaque modèle
summary(
  resamples(
    list(
      model1 = cv_model1, 
      model2 = cv_model2, 
      model3 = cv_model3
    )
  )
)$statistics


###Matrice de confusion
#prediction
pred_class1<- predict(cv_model1, df_train)
pred_class2<- predict(cv_model2, df_train)
pred_class3<- predict(cv_model3, df_train)
# create confusion matrix
conf1=confusionMatrix(
  data = relevel(pred_class1, ref = "no"), 
  reference = relevel(df_train$y, ref = "no")
)
conf1
conf2=confusionMatrix(
  data = relevel(pred_class2, ref = "no"), 
  reference = relevel(df_train$y, ref = "no")
)
conf2
conf3=confusionMatrix(
  data = relevel(pred_class3, ref = "no"), 
  reference = relevel(df_train$y, ref = "no")
)
conf3


#RANDOM FOREST
x_test1<-df_test[setdiff(names(df_test[,c("balance","duration","month","poutcome","campaign","job","contact")]),"y")]
y_test=df_test$y

rf1<-randomForest(y~balance+duration+month+poutcome+campaign+job+contact, data=df_train,
                  ntree=500,
                  MTRY=3,
                  replace=T,
                  nodesize=T,
                  keep.forest=TRUE,
                  ytest=y_test,
                  xtest=x_test1,
                  cutoff=c(0.5,0.25))
rf1

x_test<-df_test[setdiff(names(df_test),"y")]

rf2<-randomForest(y~., data=df_train,
                  ntree=500,
                  MTRY=3,
                  replace=T,
                  nodesize=T,
                  keep.forest=TRUE,
                  ytest=y_test,
                  xtest=x_test,
                  cutoff=c(0.5,0.25))
rf2

x_test3<-df_test[setdiff(names(df_test[,c("duration","poutcome","month","contact","housing","job","campaign","loan","marital","balance",
                                          "education","day","previous")]),"y")]

rf3<-randomForest(y~duration + poutcome + month + contact + housing + 
                  job + campaign + loan + marital + balance + education + day + previous 
                  ,data=df_train,
                  ntree=500,
                  MTRY=3,
                  replace=T,
                  nodesize=T,
                  keep.forest=TRUE,
                  ytest=y_test,
                  xtest=x_test3,
                  cutoff=c(0.5,0.25))
rf3


######################Mesure des performances######################

#GLM

#Calcul de l'AUC des deux meilleurs modèle GLM avec la matrice de confusion
perf.p<-cbind(df_test, pred_prob=predict(cv_model2,df_test,type="prob"))
pred<-prediction(perf.p$pred_prob.yes, df_test$y)
auc<-performance(pred,"auc")
auc@y.values[[1]]

pred_test2<- predict(cv_model2, df_test)
conf2=confusionMatrix(
  data = relevel(pred_test2, ref = "no"), 
  reference = relevel(df_test$y, ref = "no")
)
conf2

perf.p<-cbind(df_test, pred_prob=predict(cv_model3,df_test,type="prob"))
pred<-prediction(perf.p$pred_prob.yes, df_test$y)
auc<-performance(pred,"auc")
auc@y.values[[1]]

pred_test2<- predict(cv_model3, df_test)
conf2=confusionMatrix(
  data = relevel(pred_test2, ref = "no"), 
  reference = relevel(df_test$y, ref = "no")
)
conf2

#RANDOM FOREST

#Calcul de l'AUC du modèle 3 avec random forest
perf.p<-cbind(df_test, pred_prob=predict(rf3,df_test,type="prob"))
head(perf.p)
pred<-prediction(perf.p$pred_prob.yes, df_test$y)
auc<-performance(pred,"auc")
auc@y.values[[1]]




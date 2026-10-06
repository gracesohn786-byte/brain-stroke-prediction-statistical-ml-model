# Create Data
Never1 = do.call("rbind", replicate(24, c(1, 0), simplify = FALSE))
Never0 = do.call("rbind", replicate(1355, c(0, 0), simplify = FALSE))
Occasional1 = do.call("rbind", replicate(35, c(1, 2), simplify = FALSE))
Occasional0 = do.call("rbind", replicate(603, c(0, 2), simplify = FALSE))
Nearly1 = do.call("rbind", replicate(21, c(1, 4), simplify = FALSE))
Nearly0 = do.call("rbind", replicate(192, c(0, 4), simplify = FALSE))
Every1 = do.call("rbind", replicate(30, c(1, 5), simplify = FALSE))
Every0 = do.call("rbind", replicate(224, c(0, 5), simplify = FALSE))

mydata = data.frame(rbind(Never1, Never0, Occasional1, Occasional0, Nearly1, Nearly0, Every1, Every0))
# Change variable names 
names(mydata)[1] <- "Y"
names(mydata)[2] <- "X"

# Linear Probability Model
mod1 <- glm(Y~X, data = mydata, family = gaussian)
summary(mod1)
mod11 <- lm(Y~X, data = mydata)
summary(mod11)

# Logistic Regression Model
mod2 <- glm(Y~X, data = mydata, family = binomial)
summary(mod2)

# Probit Regression Model
mod3 <- glm(Y~X, data = mydata, family = binomial(link = "probit"))
summary(mod3)


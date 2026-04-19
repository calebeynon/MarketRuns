# MODEL FOR HIGH INFO
#LOW liquidity

####### PARAMETERS ######

l=0.125
mub=0.675
mug=1-mub

beliefs=c(0,0.01,0.02,0.05,0.1,0.19,0.33,0.5,0.68,0.82,0.9,0.95,0.98,0.99,1)
num_beliefs=length(beliefs)
index_b=c(1,3,4,5,6,7,8,9,10,11,12,13,14,15,15)
index_g=c(1,1,2,3,4,5,6,7,8,9,10,11,12,13,15)
prob_b=beliefs*mub+(1-beliefs)*mug
prob_g=beliefs*(1-mub)+(1-beliefs)*(1-mug)

prices=c(8,5.5,3,0.5)
v=20

tol=0.001
max_iter=50


######################################
### ONE PLAYER ###

V_1=pmax(prices[4],(beliefs*mean(prices[4])+(1-beliefs)*v))
term_value=(1-beliefs)*v+beliefs*mean(prices[4])
cont_value=prob_b*V_1[index_b]+prob_g*V_1[index_g]
W_1=l*term_value+(1-l)*cont_value



######################################

######################################
###########  TWO PLAYERS ##########

#initialize value function
V_2=pmax(prices[3],(beliefs*mean(prices[c(3,4)])+(1-beliefs)*v))
new_V_2=V_2

#strategies
x1=0.157
x2=1
x3=1
x4=1
sig=(beliefs>=0.95)*1
sig[num_beliefs-4]=x1
#sig[num_beliefs-4]=x2
#sig[num_beliefs-5]=x3
#sig[num_beliefs-6]=x4


s_space=c(0,1)
s_probs=matrix(nrow = 2,ncol = num_beliefs)
s_probs[1,]=1-sig
s_probs[2,]=sig


iter=0
go=0
while(go==0&iter<max_iter){
  iter=iter+1
  V_2=new_V_2
  
  cont_value=prob_b*V_2[index_b]+prob_g*V_2[index_g]
  
  term_value=(1-beliefs)*v + beliefs * mean(prices[c(3,4)])
  
  W_2=l*term_value+(1-l)*cont_value
  
  exp_W=colSums(s_probs*rbind(W_2,W_1))
  
  exp_price=colSums(s_probs*rbind(rep(prices[3],num_beliefs),rep(mean(prices[c(3,4)]),num_beliefs)))
  
  new_V_2=pmax(exp_price, exp_W)
  
  #distance
  dist=max(abs(new_V_2/V_2-1))
  if(dist<tol){go=1}
}

V_2=new_V_2
W_2=l*term_value+(1-l)*cont_value

print(exp_price-exp_W)
print(sig)
######################################

######################################
############# THREE PLAYERS #############

#initialize value function
V_3=pmax(prices[2],(beliefs*mean(prices[c(2,3,4)])+(1-beliefs)*v))
new_V_3=V_3
#strategies
x1=0.315
x2=1
x3=1
x4=1
sig=(beliefs>=0.90)*1
sig[num_beliefs-5]=x1
#sig[num_beliefs-4]=x2
#sig[num_beliefs-5]=x3
#sig[num_beliefs-6]=x4

s_space=c(0,1,2)
s_probs=matrix(nrow = 3,ncol = num_beliefs)
s_probs[1,]=(1-sig)^2
s_probs[2,]=2*sig*(1-sig)
s_probs[3,]=sig^2

#loop
iter=0
go=0
while(go==0&iter<max_iter){
  iter=iter+1
  V_3=new_V_3
  
  cont_value=prob_b*V_3[index_b]+prob_g*V_3[index_g]
  
  term_value=(1-beliefs)*v + beliefs * mean(prices[c(2,3,4)])
  
  W_3=l*term_value+(1-l)*cont_value
  
  exp_W=colSums(s_probs*rbind(W_3,W_2,W_1))
  
  exp_price=colSums(s_probs*rbind(rep(mean(prices[c(2)]),num_beliefs),rep(mean(prices[c(2,3)]),num_beliefs),rep(mean(prices[c(2,3,4)]),num_beliefs)))

  new_V_3=pmax(exp_price, exp_W)
  
  #distance
  dist=max(abs(new_V_3/V_3-1))
  if(dist<tol){go=1}
}

V_3=new_V_3
W_3=l*term_value+(1-l)*cont_value

print(exp_price-exp_W)
print(sig)
######################################



######################################


######################################
############# FOUR PLAYERS #############

#initialize value function
V_4=pmax(8,(beliefs*mean(prices[c(1,2,3,4)])+(1-beliefs)*v))
new_V_4=V_4
#strategies
x1=0.1298
x2=1
x3=1
x4=1
sig=(beliefs>=0.82)*1
sig[num_beliefs-6]=x1
#sig[num_beliefs-4]=x2
#sig[num_beliefs-5]=x3
#sig[num_beliefs-6]=x4

s_space=c(0,1,2,3)
s_probs=matrix(nrow = 4,ncol = num_beliefs)
s_probs[1,]=(1-sig)^3
s_probs[2,]=3*sig*(1-sig)^2
s_probs[3,]=3*sig^2*(1-sig)
s_probs[4,]=sig^3

#loop
iter=0
go=0
while(go==0&iter<max_iter){
  iter=iter+1
  V_4=new_V_4
  
  cont_value=prob_b*V_4[index_b]+prob_g*V_4[index_g]
  
  term_value=(1-beliefs)*v + beliefs * mean(prices[c(1,2,3,4)])
  
  W_4=l*term_value+(1-l)*cont_value
  
  exp_W=colSums(s_probs*rbind(W_4,W_3,W_2,W_1))
  
  exp_price=colSums(s_probs*rbind(rep(prices[c(1)],num_beliefs),rep(mean(prices[c(1,2)]),num_beliefs),rep(mean(prices[c(1,2,3)]),num_beliefs),rep(mean(prices[c(1,2,3,4)]),num_beliefs)))
  
  new_V_4=pmax(exp_price, exp_W)
  
  #distance
  dist=max(abs(new_V_4/V_4-1))
  if(dist<tol){go=1}
}

V_4=new_V_4
W_4=l*term_value+(1-l)*cont_value
print(exp_price-exp_W)
print(sig)
##########################################
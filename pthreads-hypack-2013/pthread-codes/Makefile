
CC = gcc
CFLAGS = -w 
LDFLAGS =  -lpthread -lm

TARGETS =   pthread-demo-datarace pthread-helloworld pthread-sumn pthread-join  pthread-stack  pthread-mutex  pthread-conditionvariable  pthread-numerical-integration  pthread-vectorvector-multi  pthread-infinitynorm-rowwise  pthread-infinitynorm-colwise  pthread-jacobi  pthread-findminimumvalue  pthread-producer-consumer  pthread-finding-k-matches

all: $(TARGETS) 

$(TARGETS):  %:%.o 
	$(CC) -o $@ $(CFLAGS) $^  $(LDFLAGS)
clean:
	$(RM) *.o $(TARGETS) 



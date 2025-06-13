/******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 1.5 : javathread_program_producer_consumer.java
 
  Objective   :Write a Pthread program to illustrate producer-consumer problem.

  Input       : dataObject 
		NumberOfProducers 
                NumberOfConsumers 
                ProducersPriority
                ConsumersPriority 
                SleepInterval 
                YieldFlag 

  Output      : None                                                                                                             
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

*****************************************************************/

//Producer/Consumer example. Version 0.

class MyData0
{
  protected int Data;
  protected boolean Ready;
  protected boolean Taken;
  protected boolean Yield;
  
  public void store(int Data)
  {
    this.Data=Data;
  }
  public int load()
  {
    return this.Data;
  }
}

class MyData1 extends MyData0
{
  
  public MyData1(boolean Yield)
  {
    Ready=false;
    Taken=true;
    this.Yield=Yield;
  }
  
  public void store(int Data)
  {
    while(!Taken)
      {if (Yield==true) Thread.yield();};
    this.Data=Data;
    Taken=false;
    Ready=true;
  }
 
  public int load()
  {
    int Data;
    while(!Ready)
      {if (Yield==true) Thread.yield();};
    Data=this.Data;
    Ready=false;
    Taken=true;
    return Data;
  }
}

//Producer/Consumer example. Version 2.

class MyData2 extends MyData0
{  
  public MyData2(boolean Yield)
  {
    Ready=false;
    Taken=true;
    this.Yield=Yield;
  }
  
  public synchronized void store(int Data)
  {
    while (!Taken)
      {if (Yield==true) Thread.yield();};
    this.Data=Data;
    Taken=false;
    Ready=true;
  }
  
  public synchronized int load()
  {
    int Data;
    while(!Ready)
      {if (Yield==true) Thread.yield();};
    Data=this.Data;
    Ready=false;
    Taken=true;
    return Data;
  }
}

//Producer/Consumer example. Version 3.

class MyData3 extends MyData0
{
  
  public MyData3(boolean Yield)
  {
    Ready=false;
    Taken=true;
    this.Yield=Yield;
  }
  
  public void store(int Data)
  {
    while (!Taken)
      {if (Yield==true) Thread.yield();};
    synchronized (this)
      {
	this.Data=Data;
	Taken=false;
	Ready=true;
      }
  }
  
  public int load()
  {
    while(!Ready)
      {if (Yield==true) Thread.yield();};
      synchronized (this)
      {
	Ready=false;
	Taken=true;
	return this.Data;
      }
  }
}

//Producer/Consumer example. Version 4.

class MyData4 extends MyData0
{
  
  public MyData4()
  {
    Ready=false;
  }
  
  public synchronized void store(int Data)
  {
    while (Ready)
      try 
      {
	wait();
      } catch (InterruptedException e) {}  
      this.Data=Data;
      Ready=true;
      notify();
  }
  
  public synchronized int load()
  {
    int Data;
    while(!Ready)
      try
      {
	wait();
      } catch (InterruptedException e) {}  
      Data=this.Data;
      Ready=false;
      notify();
      return Data;
  }
}

class javathread_program_producer_consumer
{
  public static void main (String[] argv)
  {
    MyData0 data;
    Thread Producers[];
    Thread Consumers[];
    int ProducersNumber;
    int ConsumersNumber;
    int producersPriority;
    int consumersPriority;
    int interval;
    boolean Yield; 
    
    if  (argv.length < 7 ) {
      System.out.print("Usage: \n> java ProducerConsumerDriver <dataObject> ");
      System.out.print("<NumberOfProducers> <NumberOfConsumers> ");
      System.out.println("<ProducersPriority> <ConsumersPriority>");
      System.out.println("<SleepInterval> <YieldFlag> ");
      return;
    }
    
    if (Integer.valueOf(argv[6]).intValue()==0) Yield=false; 
    else Yield=true; 
    
    switch(Integer.valueOf(argv[0]).intValue()){
    case 0:
      data = new MyData0(); break;
    case 1:
      data = new MyData1(Yield); break;
    case 2:
      data = new MyData2(Yield); break;
    case 3:
      data = new MyData3(Yield); break;
    case 4:
      data = new MyData4(); break;
    default: 
      data = new MyData0();
    };

    ProducersNumber=Integer.valueOf(argv[1]).intValue();
    
    ConsumersNumber=Integer.valueOf(argv[2]).intValue();
    
    producersPriority = Integer.valueOf(argv[3]).intValue();

    consumersPriority = Integer.valueOf(argv[4]).intValue();

    interval=Integer.valueOf(argv[5]).intValue();

    Producers = new Thread[ProducersNumber];
    
    Consumers = new Thread[ConsumersNumber];

    for(int i=0;i<ProducersNumber;i++){
      Producers[i]=new Thread (new Producer(data, i, interval));
      Producers[i].setPriority(producersPriority);
      Producers[i].start();
    }
    
    for(int i=0;i<ConsumersNumber;i++) {
      Consumers[i]=new Thread (new Consumer(data, i, interval));
      Consumers[i].setPriority(consumersPriority);
      Consumers[i].start();
    }
  }
}   

class Producer implements Runnable
{
  MyData0 data;
  int interval;
  int identifier;
  
  public Producer(MyData0 data, int identifier, int interval)
  {
    this.data=data;
    this.interval=interval;
    this.identifier=identifier;
  }
  
  public void run()
  {
    int i;
    for(i=0;;i++)
      {
	data.store(i);
	System.out.println("Producer "+identifier+": "+i);
	try
	  {
	    Thread.sleep((int) (Math.random()*interval));
       } catch (InterruptedException e) {System.out.println("Producer error\n");}
      }
  }
}
class Consumer implements Runnable
{
  MyData0 data;
  int interval;
  int identifier;
  
  public Consumer(MyData0 data, int identifier, int interval)
  {
    this.data=data;
    this.interval=interval;
    this.identifier=identifier;
  }
  
  public void run()
  {
    int i;
    for(;;)
      {
	System.out.println("        Consumer "+identifier+": "+data.load());
	try
	  {
	    Thread.sleep((int) (Math.random()*interval));
	  } catch (InterruptedException e) {System.out.println("Consumer error\n");}
      }
  }
}


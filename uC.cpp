#include "mbed.h"              
#include "stdio.h"
 
// Initialize a pins to perform analog input and digital output fucntions
AnalogIn   ain(p17);
DigitalOut dout(LED1);
Serial pc(USBTX,USBRX);

int main(void)
{
    pc.baud(115200);
    while (1) {
      
        if(ain > 0.15f) {
            dout = 1;
            pc.printf("%3.0f\n", ain.read()*100.0f);
         
              } 
        else {
         
            dout = 0;
        }
        wait(1);
    }
}

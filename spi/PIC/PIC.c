/*
 * File:   spi.c
 * Author: temp
 *
 * Created on November 9, 2018, 10:03 PM
 */

// 'C' source line config statements

// FBS
#pragma config BWRP = WRPROTECT_OFF     // Boot Segment Write Protect (Boot Segment may be written)
#pragma config BSS = NO_FLASH           // Boot Segment Program Flash Code Protection (No Boot program Flash segment)
#pragma config RBS = NO_RAM             // Boot Segment RAM Protection (No Boot RAM)

// FSS
#pragma config SWRP = WRPROTECT_OFF     // Secure Segment Program Write Protect (Secure segment may be written)
#pragma config SSS = NO_FLASH           // Secure Segment Program Flash Code Protection (No Secure Segment)
#pragma config RSS = NO_RAM             // Secure Segment Data RAM Protection (No Secure RAM)

// FGS
#pragma config GWRP = OFF               // General Code Segment Write Protect (User program memory is not write-protected)
#pragma config GSS = OFF                // General Segment Code Protection (User program memory is not code-protected)

// FOSCSEL
#pragma config FNOSC = FRCPLL           // Oscillator Mode (Internal Fast RC (FRC) w/ PLL)
#pragma config IESO = ON                // Internal External Switch Over Mode (Start-up device with FRC, then automatically switch to user-selected oscillator source when ready)

// FOSC
#pragma config POSCMD = NONE            // Primary Oscillator Source (Primary Oscillator Disabled)
#pragma config OSCIOFNC = ON            // OSC2 Pin Function (OSC2 pin has digital I/O function)
#pragma config IOL1WAY = ON             // Peripheral Pin Select Configuration (Allow Only One Re-configuration)
#pragma config FCKSM = CSDCMD           // Clock Switching and Monitor (Both Clock Switching and Fail-Safe Clock Monitor are disabled)

// FWDT
#pragma config WDTPOST = PS32768        // Watchdog Timer Postscaler (1:32,768)
#pragma config WDTPRE = PR128           // WDT Prescaler (1:128)
#pragma config WINDIS = OFF             // Watchdog Timer Window (Watchdog Timer in Non-Window mode)
#pragma config FWDTEN = OFF             // Watchdog Timer Enable (Watchdog timer enabled/disabled by user software)

// FPOR
#pragma config FPWRT = PWR64            // POR Timer Value (64ms)
#pragma config ALTI2C = OFF             // Alternate I2C  pins (I2C mapped to SDA1/SCL1 pins)

// FICD
#pragma config ICS = PGD1               // Comm Channel Select (Communicate on PGC1/EMUC1 and PGD1/EMUD1)
#pragma config JTAGEN = OFF             // JTAG Port Enable (JTAG is Disabled)

#include <p33Fxxxx.h>
#include <libpic30.h>
#include <xc.h>
#include <stdio.h>
#include <dsp.h>
#include <math.h>
#include <stdint.h>

// Keyword for interrupt
#define intpt __attribute__((interrupt(auto_psv)))


// Audio
int16_t ADCValue, DACValue = 0; // Audio values for buffers


// SPI
int16_t *audio;           // Oldest audio sample
int16_t *buf;             // Newest audio sample
#define buf_size 4096
int16_t buffer[buf_size]; // Buffer to store audio samples between spi chunks
uint16_t payload;         // From master

// Sets pointers back to beginning of buffer
void reset_buffer() {
    buffer[0] = 0x5555;
    audio = &buffer[1];
    buf = &buffer[0];
}


// Filters
// Keywords for storing in .xdata and .ydata
#define _YDAT(N) __attribute__((eds, space(ymemory), aligned(N)))
#define _XDAT __attribute__((eds, space(xmemory)))


// Feedback variables
uint16_t fi, fb0, fb1, fb2;   // Components of feedback vector

// Feedback Vector
struct feedback {uint16_t *data;
                 struct feedback *next;}
//I do declare
f0,
f =  {&fi, &f0},
f2 = {&fb2, &f},
f1 = {&fb1, &f2},
f0 = {&fb0, &f1};

struct feedback *response = &f0; // Points to current message
int fbFull; // Describes the status of the feedback vector

// Resets feedback vector and status
void reset_feedback(void) {
    *(f0.data) = 0;
    *(f1.data) = 0;
    *(f2.data) = 0;
    *(f.data) = 0;
    response = &f;
    fbFull = 0;
}

// Initializes Timer3 to reset feedback vecror
void init_FBTIMER(void) {

    // T1CON timer mode config
    T3CONbits.TCS = 0;      // Internal Clock
    T3CONbits.TCKPS = 0b10; // No prescaler

    PR3 = 0x0DCB; // run timer up to this value
    TMR3 = 0;

    // Interrupt clear and setup
    IPC2bits.T3IP = 1; // Priority
    IFS0bits.T3IF = 0;
    IEC0bits.T3IE = 1;

    T3CONbits.TON = 1; // Start timer
}

// Timer3 interrupt
void intpt _T3Interrupt(void) {

    reset_feedback();

    // Reset interrupt
    TMR3 = 0;
    IFS0bits.T3IF = 0;
}


// Initialized DAC module
void init_DAC(void) {
    DAC1CONbits.DACEN = 0;    // Turn off DAC module to make sure conversion doesn't start mid-config
    DAC1CONbits.DACSIDL = 0;  // Continue module operation in Idle mode (same as with ADC)

    TRISBbits.TRISB12 = 0;    // Set port B bit 12 to output
    LATBbits.LATB12 = 0;  	  // Clear

    // Set up DAC clock
    ACLKCONbits.SELACLK = 0;  // FRC w/ PLL as Clock Source 
    ACLKCONbits.AOSCMD = 0;	  // Auxiliary Oscillator Disabled
    ACLKCONbits.ASRCSEL = 1;  // Primary Oscillator is the Clock Source
    ACLKCONbits.APSTSCLR = 4; // ACLK = FCV0/3 

    DAC1STATbits.ROEN = 1;    // Enable right DAC output
    //DAC1STATbits.LOEN = 0;  // Enable left DAC output 

    DAC1DFLT = 0x0000;        // Default to midpoint output voltage 

    DAC1CONbits.AMPON = 0;    // Analog output amp disabled during Sleep/Idle mode
    DAC1CONbits.FORM = 1;     // Data format as signed integer 

    // Interrupts
    DAC1STATbits.RITYPE = 0;  // Right CH Interrupt when FIFO is not full
    //DAC1STATbits.LITYPE = 0;// LeftCH Interrupt when FIFO is not full

    IFS4bits.DAC1RIF = 0;     // Clear Right CH interrupt flag
    //IFS4bits.DAC1LIF = 0;   // Clear Left CH interrupt flag

    IEC4bits.DAC1RIE = 1;     // Right CH Interrupt Enable
    //IEC4bits.DAC1LIE = 1;   // Left CH Interrupt Enable
    
    IPC19bits.DAC1RIP = 4;// Set interrupt priority

    DAC1RDAT = 0x0000;        // Initiate DAC by writing to R&L outputs 

    DAC1CONbits.DACFDIV = 3;  // Divide DAC clock
    
    DAC1CONbits.DACEN = 1;    // Enable DAC mode
    AD1PCFGLbits.PCFG12 = 0;  // RB12 pin to Analog mode
}

void intpt _DAC1RInterrupt(void) {
    
    //LATAbits.LATA4 = !LATAbits.LATA4;
    
    // Filter
    DACValue = filter(ADCValue);
    
    // Output
    DAC1RDAT = DACValue;
    
    IFS4bits.DAC1RIF = 0;
}


// Initialized ADC module
void init_ADC(void) {
    AD1CON1bits.ADON = 0;

    AD1CHS0bits.CH0SA = 1;     // Analog pins AN0-AN8 can be selected to CH0
    AD1CHS0bits.CH0NA = 0;     // Negative input is Vref-

    AD1CSSLbits.CSS0 = 0;    // Skip input scan for analog pin AN0,AN2
    AD1CSSLbits.CSS1 = 1;    // Sets input scan for analog pin AN1
    AD1CSSLbits.CSS2 = 0;

    AD1CON3bits.SAMC = 0;     // Sample Time = 1 x TAD
    AD1CON3bits.ADRC = 0;     // selecting Conversion clock source derived from system clock
    AD1CON3bits.ADCS = 4;     // Selecting conversion clock TAD

    AD1CON1bits.AD12B = 1;    // 12-bit ADC operation
    AD1CON1bits.ADSIDL = 0;   // Continue module while in Idle mode
    AD1CON1bits.FORM = 1;     // Signed Integer output: ssss sddd dddd dddd
    AD1CON1bits.SSRC = 7;     // Manual clear SAMP bit to end sampling and start conversion
    AD1CON1bits.ASAM = 0;     // Sampling begins when SAMP bit is set
    AD1CON1bits.SAMP = 0;     // Makes sure sampling doesn't start while configuring (even though module is off)

    AD1CON2bits.VCFG = 0;     // Voltage Reference Configuration bits: set as AVdd and AVss
    AD1CON2bits.CSCNA = 0;    // Disable input scan
    AD1CON2bits.SMPI = 0;     // Selecting 1 conversion sample per interrupt
    AD1CON2bits.ALTS = 0;     // Always use MUX A input (goes with CH0SA selection)
    AD1CON2bits.BUFM = 0;     // Output as one 16-word buffer

    // Interrupt setup
    IPC3bits.AD1IP = 3; // Priority
    IFS0bits.AD1IF = 0; // Clear flag
    IEC0bits.AD1IE = 1;
    
    AD1CON1bits.ADON = 1;     //A/D converter is ON
    AD1PCFGLbits.PCFG1 = 0;    // AN1 pin to Analog mode
}

// Collects sample and updates buffer
void intpt _ADC1Interrupt(void) {
    
    ADCValue = ADC1BUF0 * 16;     // read the conversion result
    
    // Fill buffer if not full
    if (buf < &(buffer[buf_size-1]))
        buf++;
    *buf = ADCValue;
    
    IFS0bits.AD1IF = 0; // Clear flag
}

// Update ADCValue with most recent ADC input
void readADC(void) {
    AD1CON1bits.DONE = 0;
    AD1CON1bits.SAMP = 1;        // start sampling
}


// Initializes SPI module and syncs up with master
void init_SPI(void) {
    
    // Set peripheral pins to SPI
    RPINR20bits.SCK1R = 6; // SCK
    RPINR20bits.SDI1R = 7; // SDI
    RPOR4bits.RP8R = 7;    // SDO
    RPINR21bits.SS1R = 9;  // SS
    
    // Clear buffer
    SPI1BUF = 0b000000000000000;
    
    // Configure module
    SPI1CON1bits.MSTEN = 0;  // Disable master mode (enable slave))
    SPI1CON1bits.MODE16 = 1; // Enable 16 bit mode (disable 8 bit)
    SPI1CON1bits.CKE = 0;    // Transmit on clock idle to active
    SPI1CON1bits.CKP = 0;    // Idle is low
    SPI1CON1bits.SSEN = 1;   // Enable slave select pin4
    
    // Slave mode requirements
    SPI1CON1bits.SMP = 0; // Clear SPI Data Input Sample Phase bit
    SPI1STATbits.SPIROV = 0; // Clear SPI overflow bit
    
    // Interrupt clear and setup
    IPC2bits.SPI1IP = 1; // Priority
    IFS0bits.SPI1IF = 0;
    IEC0bits.SPI1IE = 1;
    
    // Enable SPI1 module
    SPI1STATbits.SPIEN = 1;
}

// Interrupt based SPI send/receive=
void intpt _SPI1Interrupt(void) {
    
    // Receive
    payload = SPI1BUF;
    
    // Send Audio and increment if audio behind buffer
    if (audio <= buf) {
        
        // Don't accidentally send filler
        if ( ((0x00FF&*audio) ^ 0x0055) == 0 )
            *audio = *audio & 0xFFFE;
        if ( ((0xFF00&*audio) ^ 0x5500) == 0 )
            *audio = *audio & 0xFEFF;
        
        SPI1BUF = *audio;
        audio++;
    }
    // Send Filler if audio ahead of buffer
    else {
        SPI1BUF = buffer[0];
        reset_buffer();
    }
    
    // Check for feedback response and populate vector if not full
    *(response->data) = payload;
    if ( !fbFull && ((0x003F & *(f.data)) == 0x003F) ) {
        response = response->next;
    }
    
    // Reset interrupt
    IFS0bits.SPI1IF = 0;
}


// Initializes Timer to time for sampling rate
void init_TIMER(void) {
    
    // T1CON timer mode config
    T1CONbits.TCS = 0;      // Internal Clock
    T1CONbits.TCKPS = 0b00; // No prescaler
    
    PR1 = 0x07B6; // run timer up to this value
    TMR1 = 0;
    
    // Interrupt clear and setup
    IPC0bits.T1IP = 2; // Priority
    IFS0bits.T1IF = 0;
    IEC0bits.T1IE = 1;
    
    T1CONbits.TON = 1; // Start timer
}

// Timer interrupt
//int16_t test = 0;
void intpt _T1Interrupt(void) {
    
    // Timer oscillation
    //LATBbits.LATB5 = !LATBbits.LATB5;
    
    // Reset interrupt
    TMR1 = 0;
    IFS0bits.T1IF = 0;
    
    // Read audio
    readADC();
    
    // Testing
    //test++;
    //if ((test & 0x5500) == 0x5500)
    //     test = 0;
}


// Set up clock
void init_CLK(void) {
    // Configure PLL prescaler, PLL postscaler, PLL divisor
    PLLFBDbits.PLLDIV = 47;  // M
    CLKDIVbits.PLLPOST = 0;   // N2
    CLKDIVbits.PLLPRE = 0;    // N1
    
    // Initiate Clock Switch to Internal FRC with PLL (NOSC = 0b001)
    __builtin_write_OSCCONH(0x01);
    __builtin_write_OSCCONL(OSCCON | 0x01);
    
    // Wait for Clock switch to occur
    while (OSCCONbits.COSC != 0b001);
    // Wait for PLL to lock
    while(OSCCONbits.LOCK!=1) {};
}


// Notch filter
int16_t LPF550Hz(int16_t NewSample) {
    
    static int32_t y[3]; //output samples
    static int16_t x[3]; //input samples
    
    x[2] = x[1];
    x[1] = x[0];
    x[0] = NewSample;
    y[2] = y[1];
    y[1] = y[0];
    
    y[0] = (int32_t) 389 * x[0];
    y[0] += (int32_t) - 382 * x[1] + 30669 * y[1];
    y[0] += (int32_t) 389 * x[2] - 14512 * y[2];
    y[0] /= 16384;
    
    return (y[0] / 2);
}
int16_t HPF1k2Hz(int16_t NewSample) {
    
    static int32_t y[3]; //output samples
    static int16_t x[3]; //input samples
    
    x[2] = x[1];
    x[1] = x[0];
    x[0] = NewSample;
    y[2] = y[1];
    y[1] = y[0];
    
    y[0] = (int32_t) 15284 * x[0];
    y[0] += (int32_t) - 30569* x[1] + 30440 * y[1]; 
    y[0] += (int32_t) 15284 * x[2] - 15284 * y[2];
    y[0] /= 16500;
    
    return (y[0] / 1);
}
int16_t notch(int16_t NewSample) {
    int16_t filteredValue = LPF550Hz(NewSample);
    filteredValue += HPF1k2Hz(NewSample);
    return(filteredValue);
}

// Highpass
int16_t HPF250Hz(int16_t NewSample) {
    
    static int32_t y[3]; //output samples
    static int16_t x[3]; //input samples
    
    x[2] = x[1];
    x[1] = x[0];
    x[0] = NewSample;
    y[2] = y[1];
    y[1] = y[0];
    
    y[0] = (int32_t) 15620 * x[0];
    y[0] += (int32_t) - 31241 * x[1] + 31179 * y[1];
    y[0] += (int32_t) 15620 * x[2] - 14920 * y[2];
    y[0] /= 18384;
    
    return (y[0] / 1);
}

// Lowpass
int16_t LPF5kHz(int16_t NewSample) {
    
    static int32_t y[3]; //output samples
    static int16_t x[3]; //input samples
    
    x[2] = x[1];
    x[1] = x[0];
    x[0] = NewSample;
    y[2] = y[1];
    y[1] = y[0];
    
    y[0] = (int32_t) 9799 * x[0];
    y[0] += (int32_t) 19598 * x[1] + 21263 * y[1]; 
    y[0] += (int32_t) 9799 * x[2] - 8094 * y[2];
    y[0] /= 34768;
    
    return (y[0] / 2);
}


// Filter init
int hp, lp, ntch = 0;
int16_t hpm[3] =   {0xE000, 0x0000, 0x0000};
int16_t lpm[3] =   {0x0000, 0x000F, 0xFFC0};
int16_t ntchm[3] = {0x1F80, 0x0000, 0x0000};
int i;

// Filter logic
int16_t filter(int16_t NewSample) {
    
    int16_t filteredValue = NewSample;
    
    // Highpass
    if (hp)
        filteredValue = HPF250Hz(filteredValue);
    if (lp)
        filteredValue = LPF5kHz(filteredValue);
    if (ntch)
        filteredValue = notch(filteredValue);
    
    return filteredValue;
}

// Reset filters interrupt setup
void init_RESET(void) {
    
    RPINR1bits.INT2R = 3; // RP3
    TRISBbits.TRISB3 = 1; // input
    AD1PCFGLbits.PCFG5 = 1; //digital
    
    INTCON2bits.INT2EP = 1; // Trigger on negative edge
    IFS1bits.INT2IF = 0;    // Reset flag
    IPC7bits.INT2IP = 1;    // low priority reset
    IEC1bits.INT2IE = 1;    // Enable interrupt
}

// Reset filters interrupt
void intpt _INT2Interrupt(void) {
    
    // Reset filters
    hp = 0;
    lp = 0;
    ntch = 0;
    
    IFS1bits.INT2IF = 0; // Reset flag
}


void main() {     
    
    // Initialize audio buffer
    reset_buffer();
    
    // Initialize all modules
    init_CLK();
    init_SPI();
    init_ADC();
    init_TIMER();
    init_FBTIMER();
    init_DAC();
    init_RESET();
    
    // Output pin for testing
    //TRISBbits.TRISB5 = 0;
    //LATBbits.LATB5 = 0;
    //TRISAbits.TRISA4 = 0;
    //LATAbits.LATA4 = 0;
    
    // Main loop
    reset_feedback();
    while (1) {
        
        // Check if feedback exists
        fbFull = !((0x003F & *(f2.data)) ^ 0x003F);
        
        // Ensure feedback vector footer integrity
        if ( ((0x003F & *(f2.data)) != 0x003F) && (response == &f) )
            reset_feedback();
        
        // Set filters based on feedback vector
        if (fbFull) {
            
            // highpass
            if ( (hpm[0] & *(f0.data))
              || (hpm[1] & *(f1.data))
              || (hpm[2] & *(f2.data)) )
                hp = 1;
            // lowpass
            if ( (lpm[0] & *(f0.data))
              || (lpm[1] & *(f1.data))
              || (lpm[2] & *(f2.data)) )
                lp = 1;
            // Notch
            if ( (ntchm[0] & *(f0.data))
              || (ntchm[1] & *(f1.data))
              || (ntchm[2] & *(f2.data)) )
                ntch = 1;
        }
        
    }
}
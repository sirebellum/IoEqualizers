/*
 * File:   spi.c
 * Author: temp
 *
 * Created on November 9, 2018, 10:03 PM
 */

// 'C' source line config statements

// FOSC
#pragma config FOSFPR = FRC_PLL8        // Oscillator (FRC w/PLL 8x)
#pragma config FCKSMEN = CSW_FSCM_OFF   // Clock Switching and Monitor (Sw Disabled, Mon Disabled)

// FWDT
#pragma config FWPSB = WDTPSB_16        // WDT Prescaler B (1:16)
#pragma config FWPSA = WDTPSA_512       // WDT Prescaler A (1:512)
#pragma config WDT = WDT_OFF            // Watchdog Timer (Disabled)

// FBORPOR
#pragma config FPWRT = PWRT_64          // POR Timer Value (64ms)
#pragma config BODENV = BORV20          // Brown Out Voltage (Reserved)
#pragma config BOREN = PBOR_ON          // PBOR Enable (Enabled)
#pragma config MCLRE = MCLR_EN          // Master Clear Enable (Enabled)

// FGS
#pragma config GWRP = GWRP_OFF          // General Code Segment Write Protect (Disabled)
#pragma config GCP = CODE_PROT_OFF      // General Segment Code Protection (Disabled)

// FICD
#pragma config ICS = ICS_PGD            // Comm Channel Select (Use PGC/EMUC and PGD/EMUD)

#define __dsPIC30F3012__
#include <p30Fxxxx.h>
#include <libpic30.h>
#include <xc.h>
#include <stdio.h>
#include <dsp.h>
#include <math.h>
#include <stdint.h>

// Keyword for interrupt
#define intpt __attribute__((interrupt(auto_psv)))

// GLOBAL VARIABLES:
// ADC
int16_t ADCValue = 0; // Value pulled from ADC Buffer

// SPI Transmission Content
int16_t *audio;         // Oldest audio sample
int16_t *buf;           // Newest audio sample
#define buf_size 700
int16_t buffer[buf_size];    // Buffer to store audio samples between spi chunks
uint16_t payload;       // From master

// Sets pointers back to beginning of buffer
void reset_buffer() {
    audio = &buffer;
    buf = &buffer;
}

// Feedback variables
uint16_t fb0, fb1, fb2;   // Components of feedback vector

// Feedback Vector
struct feedback {uint16_t *data;
                 struct feedback *next;}
//I do declare
f0,
f2 = {&fb2, &f0},
f1 = {&fb1, &f2},
f0 = {&fb0, &f1};

struct feedback *response = &f0; // Points to current message
int fbFull; // Describes the status of the feedback vector

// Resets feedback vector and status
void reset_feedback(void) {
    *(f0.data) = 0;
    *(f1.data) = 0;
    *(f2.data) = 0;
    fbFull = 0;
}

void init_ADC(void) {
    ADCON1bits.ADON = 0;
    
    ADCHSbits.CH0SA = 1;    // Analog pins AN0-AN8 can be selected to CH0
    ADCHSbits.CH0NA = 0;
                     
    ADCSSL = 0;              // Skip input scan for analog pins
    ADCSSLbits.CSSL1 = 1;    // Sets input scan for analog pin AN1
     
    ADCON3bits.SAMC = 0;     // Sample Time = 1 x TAD
    ADCON3bits.ADRC = 0;     // selecting Conversion clock source derived from system clock
    ADCON3bits.ADCS = 9;     // Selecting conversion clock TAD
    
    ADCON1bits.ADSIDL = 0;   // Continue module while in Idle mode -> maybe this is what's messing with SPI?
    ADCON1bits.FORM = 1;     // Signed Integer output: ssss sddd dddd dddd
    ADCON1bits.SSRC = 0;     // Manual clear SAMP bit to end sampling and start conversion
    ADCON1bits.ASAM = 0;     // Sampling begins when SAMP bit is set
    ADCON1bits.SAMP = 0;     // Makes sure sampling doesn't start while configuring (even though module is off)
      
    ADCON2bits.VCFG = 0;     // Voltage Reference Configuration bits: set as AVdd and AVss
    ADCON2bits.CSCNA = 0;    // Disable input scan
    ADCON2bits.SMPI = 0;     // Selecting 1 conversion sample per interrupt
    ADCON2bits.ALTS = 0;     // Always use MUX A input (goes with CH0SA selection)
    ADCON2bits.BUFM = 0;     // Output as one 16-word buffer
      
    ADCON1bits.ADON = 1;     //A/D converter is ON  
    ADPCFGbits.PCFG1 = 0;    // AN1 pin to Analog mode
}

void readADC(void) {                      
    ADCON1bits.SAMP = 1;  // start sampling
    __delay32(10);
    ADCON1bits.SAMP = 0;        // ends sampling, start conversion
    while ( !ADCON1bits.DONE ); // wait to complete the conversion
        ADCValue = ADCBUF0;     // read the conversion result 
}


// Initializes SPI module and syncs up with master
void init_SPI(void) {
    
    // Disable analog inputs/outputs
    ADPCFGbits.PCFG5 = 1;
    ADPCFGbits.PCFG4 = 1;
    ADPCFGbits.PCFG2 = 1;
    ADPCFGbits.PCFG6 = 1;
    
    // Clear buffer
    SPI1BUF = 0b000000000000000;
    
    // Configure module
    SPI1CONbits.MSTEN = 0;  // Disable master mode (enable slave))
    SPI1CONbits.MODE16 = 1; // Enable 16 bit mode (disable 8 bit)
    SPI1CONbits.CKE = 0;    // Transmit on clock idle to active
    SPI1CONbits.CKP = 0;    // Idle is low
    SPI1CONbits.SSEN = 1;   // Enable slave select pin4
    
    // Slave mode requirements
    SPI1CONbits.SMP = 0; // Clear SPI Data Input Sample Phase bit
    SPI1STATbits.SPIROV = 0; // Clear SPI overflow bit
    
    // Interrupt clear and setup
    IPC2bits.SPI1IP = 0b001; // Priority
    IFS0bits.SPI1IF = 0;
    IEC0bits.SPI1IE = 1;
    
    // Enable SPI1 module
    SPI1STATbits.SPIEN = 1;
}

// Interrupt based SPI send/receive
void intpt _SPI1Interrupt(void) {
    
    // Receive
    payload = SPI1BUF;
    
    // Send if audio behind buffer
    if (audio < buf) {
        
        if (*audio & 0x0055 == 0x0055) // Don't send filler
            *audio = *audio + 1;
        
        SPI1BUF = *audio;
        audio++;
    }
    // Else send filler and reset buffer
    else {
        SPI1BUF = 0x5555;
        reset_buffer();
    }
    
    // Check for feedback response and populate vector if not full
    if ((payload != 0xAAAA) && !fbFull) {
        *(response->data) = payload;
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
    
    PR1 = 0x012A; // run timer up to this value
    TMR1 = 0;
    
    // Interrupt clear and setup
    IPC0bits.T1IP = 0b010; // Priority
    IFS0bits.T1IF = 0;
    IEC0bits.T1IE = 1;
    
    T1CONbits.TON = 1; // Start timer
}

// Timer interrupt
int8_t value = 0;
int16_t test = -2048;
void intpt _T1Interrupt(void) {
    
    // Timer oscillation
    value = !value;
    LATBbits.LATB7 = value;
    TMR1 = 0;
    
    // Read audio
    readADC();
    //test++;
    //if (test >= 2048) test = -2048;
    
    // Fill buffer if not full
    *buf = ADCValue;
    if (buf < &(buffer[buf_size-1]))
        buf++;
    
    // Set filters based on feedback vector
    if (fbFull) {
        // EQ filter settings
    }
    
    // Reset interrupt
    IFS0bits.T1IF = 0;
}


void main() {     
    
    // Initialize audio buffer
    reset_buffer();
    
    // Initialize all modules
    init_SPI();
    init_ADC();
    init_TIMER();
    
    // Output pin for testing
    ADPCFGbits.PCFG7 = 1;
    TRISBbits.TRISB7 = 0;
    LATBbits.LATB7 = 0;
    
    // Main loop
    reset_feedback();
    while (1) {
        
        // Ensure feedback vector integrity
        if (0x003F & *(f2.data) != 0x003F
                  && *(f2.data) != 0)
            reset_feedback();
        
        // Check if feedback exists
        fbFull = !((0x003F & *(f2.data)) ^ 0x003F);
    }
}
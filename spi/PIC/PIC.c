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

// GLOBAL VARIABLES:
// ADC
int16_t ADCValue = 0; // Value pulled from ADC Buffer

// SPI Transmission Content
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
    AD1CON3bits.ADCS = 6;     // Selecting conversion clock TAD

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

    AD1CON1bits.ADON = 1;     //A/D converter is ON
    AD1PCFGLbits.PCFG1 = 0;    // AN1 pin to Analog mode
}

// Update ADCValue with most recent ADC input
void readADC(void) {
    AD1CON1bits.DONE = 0;
    AD1CON1bits.SAMP = 1;        // start sampling

    while ( !AD1CON1bits.DONE ); // wait to complete the conversion (about 14 x TAD))
        ADCValue = ADC1BUF0;     // read the conversion result
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
    
    // Send Audio and increment if audio behind buffer
    if (audio <= buf) {
        
        // Don't accidentally send filler
        if ( ((0x00FF&*audio) ^ 0x0055) == 0 )
            *audio = *audio + 1;
        
        SPI1BUF = *audio;
        audio++;
    }
    // Send Filler if audio ahead of buffer
    else {
        SPI1BUF = buffer[0];
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
    
    PR1 = 0x02B0; // run timer up to this value
    TMR1 = 0;
    
    // Interrupt clear and setup
    IPC0bits.T1IP = 0b010; // Priority
    IFS0bits.T1IF = 0;
    IEC0bits.T1IE = 1;
    
    T1CONbits.TON = 1; // Start timer
}

// Timer interrupt
int8_t value = 0;
//int16_t test = 0;
void intpt _T1Interrupt(void) {
    
    // Timer oscillation
    value = !value;
    LATBbits.LATB5 = value;
    
    // Reset interrupt
    TMR1 = 0;
    IFS0bits.T1IF = 0;
    
    // Read audio
    readADC();
    
    // Testing
    //test++;
    //if ((test & 0x5500) == 0x5500)
    //     test = 0;
    
    // Fill buffer if not full
    if (buf < &(buffer[buf_size-1]))
        buf++;
    *buf = ADCValue;
    
    // Set filters based on feedback vector
    if (fbFull) {
        // EQ filter settings
    }
}


void main() {     
    
    // Faster clock
    PLLFBDbits.PLLDIV = 200; // PLL Multiplier
    
    // Initialize audio buffer
    reset_buffer();
    
    // Initialize all modules
    init_SPI();
    init_ADC();
    init_TIMER();
    
    // Output pin for testing
    TRISBbits.TRISB5 = 0;
    LATBbits.LATB5 = 0;
    
    // Main loop
    reset_feedback();
    while (1) {
        
        // Ensure feedback vector footer integrity
        if ((0x003F & *(f2.data)) != 0x003F
                   && *(f2.data)  != 0)
            reset_feedback();
        
        // Check if feedback exists
        fbFull = !((0x003F & *(f2.data)) ^ 0x003F);
    }
}
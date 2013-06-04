/*
 * =====================================================================================
 *
 *       Filename:  example.c
 *
 *    Description:  Examle file to benchmark fsom library.
 *
 *        Version:  0.1
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  BlackLight (http://0x00.ath.cx), <blacklight@autistici.org>
 *    Contributor:  evilsocket (http://www.evilsocket.net), <evilsocket@gmail.com>
 *        Licence:  GNU GPL v.3
 *        Company:  DO WHAT YOU WANT CAUSE A PIRATE IS FREE, YOU ARE A PIRATE!
 *
 * =====================================================================================
 */
#include	"fsom.h"

#include	<alloca.h>
#include	<stdio.h>
#include 	<sys/time.h>

static unsigned long prev_uticks = 0;

static unsigned long get_uticks(){
    struct timeval ts;
    gettimeofday(&ts,0);
    return ((ts.tv_sec * 1000000) + ts.tv_usec);
}

static void start_timer(){
	prev_uticks = get_uticks();
}

static void print_timing( const char *msg ){
	#define MS_DELTA (1000.0)
	#define SS_DELTA (MS_DELTA * 1000.0)
	#define MM_DELTA (SS_DELTA * 60.0)
	#define HH_DELTA (MM_DELTA * 60.0)

	double ticks = get_uticks() - prev_uticks;

	if( ticks < MS_DELTA ){
	   printf( "%s\t : %lf us\n", msg, ticks );
	}
	else if( ticks < SS_DELTA ){
	   printf( "%s\t : %lf ms\n", msg, ticks / MS_DELTA );
	}
	else if( ticks < MM_DELTA ){
	   printf( "%s\t : %lf s\n", msg, ticks / SS_DELTA );
	}
	else if( ticks < HH_DELTA ){
	   printf( "%s\t : %lf m\n", msg, ticks / MM_DELTA );
	}
	else{
	   printf( "%s\t : %lf h\n", msg, ticks / HH_DELTA );
	}

	start_timer();
}

#define 	VECTORS 		10
#define 	INPUTS 		    20
#define 	OUT_ROWS 		20
#define 	OUT_COLS 		20
#define 	TRAIN_STEPS 	100

int
main ( int argc, char *argv[] )
{
	size_t i, j, x, y;
	double step = 0.0;
	double **data = NULL;
	som_network_t *net = NULL;

	data = (double **) alloca ( INPUTS * sizeof ( double* ));

	for ( i=0, step = 0.0; i < INPUTS; ++i, step += 0.1 )
	{
		data[i] = (double *) alloca ( VECTORS * sizeof ( double ));

		for ( j=0; j < VECTORS; ++j )
		{
                        data[i][j] = step;
		}
	}

	start_timer();

	net = som_network_new ( INPUTS, OUT_ROWS, OUT_COLS );

	if ( !net )
	{
		printf( "ERROR: som_network_new failed.\n" );
		return 1;
	}

	print_timing( "Network Creation" );

	som_init_weights ( net, data, INPUTS );
	som_train ( net, data, VECTORS, TRAIN_STEPS );

	print_timing( "Network Training" );

	for ( i=0; i < INPUTS; ++i )
	{
		som_set_inputs ( net, data[i] );
		som_get_best_neuron_coordinates ( net, &x, &y );
		printf ( "best coordinates [%u]: %u,%u\n", i, x, y );
	}

	print_timing( "Input Recognition" );

	som_network_destroy ( net );

	print_timing( "Network Destruction" );

	return 0;
}			

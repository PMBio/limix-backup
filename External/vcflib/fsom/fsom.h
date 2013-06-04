/*
 * =====================================================================================
 *
 *       Filename:  neural_som.h
 *
 *    Description:  Header file for neural_som mini-library
 *
 *        Version:  0.1
 *        Created:  15/10/2010 15:31:50
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

#ifndef 	__NEURAL_SOM_H
#define 	__NEURAL_SOM_H

#include	<stddef.h>
#include	<time.h>

#define TAYLOR_LAMBERT_ELEMENTS 	1001
#define TAYLOR_LAMBERT_LAST_ELEMENT 1000

#ifndef INLINE
#	define INLINE __inline__ __attribute__((always_inline))
#endif

typedef struct  {
	double                 output;
	double                 input;

	struct som_synapsis_s  **synapses;
	size_t                 synapses_count;
} som_neuron_t
__attribute__ ((aligned));;

typedef struct som_synapsis_s  {
	som_neuron_t    *neuron_in;
	som_neuron_t    *neuron_out;
	double          weight;
} som_synapsis_t
__attribute__ ((aligned));

typedef struct  {
	som_neuron_t    **neurons;
	size_t          neurons_count;
} som_input_layer_t
__attribute__ ((aligned));

typedef struct  {
	som_neuron_t    ***neurons;
	size_t          neurons_rows;
	size_t          neurons_cols;
} som_output_layer_t
__attribute__ ((aligned));

#ifdef __APPLE__
typedef struct somename {
#else
typedef struct {
#endif
	som_input_layer_t   *input_layer;
	som_output_layer_t  *output_layer;
	double              T_learning_param;
	time_t              serialization_time;
	double				alphas[TAYLOR_LAMBERT_ELEMENTS];
	double			    mus[TAYLOR_LAMBERT_ELEMENTS];
} som_network_t
__attribute__ ((aligned));

void                 som_network_destroy ( som_network_t* );
void                 som_set_inputs ( som_network_t*, double* );
void                 som_train ( som_network_t*, double**, size_t, size_t );
void                 som_serialize ( som_network_t*, const char* );
void                 som_init_weights ( som_network_t*, double**, size_t );
double               som_get_best_neuron_coordinates ( som_network_t*, size_t*, size_t* );
som_network_t*       som_deserialize ( const char* );
som_network_t*       som_network_new ( size_t, size_t, size_t );

#endif


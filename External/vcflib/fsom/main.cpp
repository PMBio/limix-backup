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
#include "fsom.h"

#include "convert.h"
#include "split.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

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
	fprintf(stderr, "%s\t : %lf us\n", msg, ticks );
    }
    else if( ticks < SS_DELTA ){
	fprintf(stderr, "%s\t : %lf ms\n", msg, ticks / MS_DELTA );
    }
    else if( ticks < MM_DELTA ){
	fprintf(stderr, "%s\t : %lf s\n", msg, ticks / SS_DELTA );
    }
    else if( ticks < HH_DELTA ){
	fprintf(stderr, "%s\t : %lf m\n", msg, ticks / MM_DELTA );
    }
    else{
	fprintf(stderr, "%s\t : %lf h\n", msg, ticks / HH_DELTA );
    }

    start_timer();
}


void printSummary(char** argv) {
    cerr << "usage: " << argv[0] << " [options]" << endl
         << endl
         << "Trains and applies a self-organizing map to the input tab-separated data on" << endl
	 << "stdin, adding two columns for the x and y coordinates of the winning neuron" << endl
	 << "in the network." << endl
	 << endl
	 << "If a map is provided, training will be skipped and the map will be applied to" << endl
	 << "the input.  Maps may be saved for later use." << endl
	 << endl
         << "options:" << endl
         << "    -h, --help          this dialog" << endl
	 << "    -a, --apply FILE    apply the saved map to input data to FILE" << endl
	 << "    -s, --save  FILE    train on input data and save the map to FILE" << endl
         << "    -x, --width X       width in columns of the output array" << endl
         << "    -y, --height Y      height in columns of the output array" << endl
         << "    -i, --iterations N  number of training iterations or epochs" << endl;
}


int main(int argc, char** argv) {   

    int width = 100;
    int height = 100;
    int num_dimensions = 2;
    int iterations = 1000;
    string save_file;
    string load_file;

    int c;

    if (argc == 1) {
        printSummary(argv);
        exit(1);
    }

    while (true) {
        static struct option long_options[] =
        {  
            /* These options set a flag. */
            //{"verbose", no_argument,       &verbose_flag, 1},
            {"help", no_argument, 0, 'h'},
            {"iterations", required_argument, 0, 'i'},
            {"width", required_argument, 0, 'x'},
            {"height", required_argument, 0, 'y'},
	    {"apply", required_argument, 0, 'a'},
	    {"save", required_argument, 0, 's'},
            {0, 0, 0, 0}
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long (argc, argv, "hi:x:y:a:s:",
                         long_options, &option_index);

        if (c == -1)
            break;

        string field;

        switch (c)
        {

            case 'x':
                if (!convert(optarg, width)) {
                    cerr << "could not parse --width, -x" << endl;
                    exit(1);
                }
                break;

            case 'y':
                if (!convert(optarg, height)) {
                    cerr << "could not parse --height, -y" << endl;
                    exit(1);
                }
                break;

            case 'i':
                if (!convert(optarg, iterations)) {
                    cerr << "could not parse --iterations, -i" << endl;
                    exit(1);
                }
                break;

            case 'a':
                load_file = optarg;
                break;

            case 's':
                save_file = optarg;
                break;

            case 'h':
                printSummary(argv);
                exit(0);
                break;

            default:
                break;
        }
    }

    size_t i, j;
    som_network_t *net = NULL;
    vector<string> inputs;
    vector<vector<double> > data;


    string line;
    stringstream ss;

    /*
    if (!load_file.empty()) { // apply
	while (getline(cin, line)) {
	    vector<double> record;
	    vector<string> items;
	    split(line, '\t', items);
	    int ti;
	    vector<string>::iterator j = items.begin();
	    for (; j != items.end(); ++j) {
		convert(*j, ti);
		record.push_back(ti);
	    }
	    som_set_inputs(net, &record[0]);
	    som_get_best_neuron_coordinates(net, &x, &y);
	    cout << x << "\t" << y << "\t" << line << endl;
	}
	return 0; // done with application
    } else { // get data
    */
    while (getline(cin, line)) {
	inputs.push_back(line);
	vector<double> record;
	vector<string> items;
	split(line, '\t', items);
	double td;
	vector<string>::iterator j = items.begin();
	for (; j != items.end(); ++j) {
	    convert(*j, td);
	    record.push_back(td);
	}
	data.push_back(record);
    }
    //}

    vector<double*> dataptrs (data.size());
    for (unsigned i=0, e=dataptrs.size(); i<e; ++i) {
	dataptrs[i] = &(data[i][0]); // assuming !thing[i].empty()
    }

    start_timer();

    if (!load_file.empty()) {
	cerr << "Loading ... "  << endl;
	if (! (net = som_deserialize(load_file.c_str()))) {
	    cerr << "could not load SOM from " << load_file << endl;
	    return 1;
	}
    } else {

	net = som_network_new(data[0].size(), height, width);
	
	if ( !net )	{
	    printf( "ERROR: som_network_new failed.\n" );
	    return 1;
	}
    }

    print_timing( "Network Creation" );

    if (!save_file.empty()) {
	cerr << "Training using " << data.size() << " input vectors" << endl;
	som_init_weights ( net, &dataptrs[0], data.size() );
	som_train ( net, &dataptrs[0], data.size(), iterations );
    }

    print_timing( "Network Training" );

    for ( i=0; i < data.size(); ++i ) {
	som_set_inputs ( net, dataptrs[i] );
	size_t x=0, y=0;
	som_get_best_neuron_coordinates ( net, &x, &y );
	//printf ( "best coordinates [%u]: %u,%u\n", i, x, y );
	cout << x << "\t" << y << "\t" << inputs[i] << endl;
    }

    print_timing( "Input Recognition" );

    if (!save_file.empty()) {
	som_serialize(net, save_file.c_str());
    }

    som_network_destroy ( net );

    print_timing( "Network Destruction" );

    return 0;
}			

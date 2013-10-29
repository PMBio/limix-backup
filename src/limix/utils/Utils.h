#if !defined( Utils_h )
#define Utils_h
/*
 *******************************************************************
 *
 *    Copyright (c) Microsoft. All rights reserved.
 *    This code is licensed under the Apache License, Version 2.0.
 *    THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 *    ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 *    IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 *    PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
 *
 ******************************************************************
 */

/*
 * Utils - {Utility Routines}
 *
 *         File Name:   Utils.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *    Module Purpose:   This file declares a number of utility 
 *                      routines for console applications
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * 'Publish' our class declarations / function prototypes
 */
// Console I/O Utility Routines
void Error( const char *szFmt, ... );
void Fatal( const char *szFmt, ... );
void FatalIfPreviousErrors();
void NTLastError( void );
void Progress( const char *szFmt, ... );
void ProgressNL( const char *szFmt, ... );
void Verbose( const char *szFmt, ... );
void Warn( const char *szFmt, ... );

// File / File system Utility Routines
int    FFileExists( const string& name );
int    FDirExists( const string& name );
bool   FIsFilenameWritable( const string& filename );
string FullPath( const string& filename );
void   MakeDirectory( const string& dirname );
string MakePath( const string& drive, const string& dir, const string& fname, const string& ext );
void   SplitPath( const string& filename, string& drive, string& dir, string& fname, string& ext );
bool   FIsCsvFile( const string& fname );    // does the filename end in '.csv'

/*
 * 'Publish' the globals we define
 */
extern int fVerbose;
extern int fReportTimes;

const int Tab = '\t';
const int NewLine = '\n';

/*
 *  Define the stubs, translation routines, and 'stuff'
 *   that Windows normally provides so we can support 
 *   a bit of cross platform operation
 */
#if !defined( _MSC_VER )
#include <linux/limits.h>

inline real abs( real x ) { return( fabs(x) ); }
inline int min( int m1, int m2 ) { return( ( m1 < m2 ) ? m1 : m2 ); }
inline real min( real m1, real m2 ) { return( (m1 < m2 ) ? m1 : m2 ); }
inline int max( int m1, int m2 ) { return( ( m1 > m2 ) ? m1 : m2 ); }
inline int _strcmpi( const char *s1, const char *s2 ) { return( strcasecmp( s1, s2 ) ); }
inline int _access( const char *path, int mode ) { return( access( path, mode ) ); }
inline long long _ftelli64( FILE *pfile ) { return( (long long)(ftell( pfile )) ); }
inline int _fseeki64( FILE *pfile, long long offset, int origin ) { return( fseek( pfile, offset, origin ) ); }
inline int QueryPerformanceCounter( LARGE_INTEGER* lpPerformanceCount ) { *lpPerformanceCount = (long long) clock(); return( 0 ); }
inline int _mkdir( char *szDir ) { return( mkdir( szDir, 0777 ) ); }
inline int QueryPerformanceFrequency( LARGE_INTEGER* lpFrequency ) { *lpFrequency = CLOCKS_PER_SEC; return( 0 ); }

int  GetComputerName( char *pchName, unsigned long* pcchMax );
#define  MAX_COMPUTERNAME_LENGTH 15

//char *_fullpath( char *absPath, const char *relPath, size_t maxLength );
#define  _MAX_PATH   260

unsigned int _set_output_format( unsigned int format );
#define  _TWO_DIGIT_EXPONENT  1
#endif   // ! _MSC_VER

/*
 *  ColumnMajor means individual column elements are located together in linear
 *    memory.  For a 10x10 matrix:
 *       [row,col] 
 *           [0,0] =>  0  (keep all column 0 together in memory)
 *           [1,0] =>  1
 *           [2,0] =>  2
 *           [0,1] => 10
 */
inline size_t ColumnMajorIndex( size_t cRows, size_t cColumns, size_t iRow, size_t iColumn )
   {
   size_t index = (cRows*iColumn) + iRow;
   return( index );
   }

/*
 *  RowMajor means individual row elements are located together in linear
 *    memory.  For a 10x10 matrix:
 *       [row,col] 
 *           [0,0] =>  0  (keep all row 0 together in memory)
 *           [0,1] =>  1
 *           [0,2] =>  2
 *           [1,0] => 10
 */
inline size_t RowMajorIndex( size_t cRows, size_t cColumns, size_t iRow, size_t iColumn )
   {
   size_t index = (cColumns*iRow) + iColumn;
   return( index );
   }

inline void ZeroRealArray( real *pReal, size_t cReals )
   {
   for ( size_t i=0; i<cReals; ++i )
      {
      *pReal = 0.0;
      ++pReal;
      }
   }

class CTimer
   {
public:
   CTimer() { Init( false ); }
   CTimer( bool fStart ) { Init( fStart ); }
   CTimer( const CTimer& src );
   void Start( bool fRestart = false );
   void Stop();
   bool IsRunning();    // 
   double Elapsed();    // elapsed time in seconds
   double Elapsedms();  // elapsed time in milliseconds
   double Elapsedus();  // elapsed time in microseconds
   double Resolution() { return(1.0/(double)m_frequency); } // timer resolution in seconds.
   string ToString();                                    // returns a formatted string with the elapsed time
   string ToString( const string& formatString );        // returns formated string
   void   Report( const string& formatString );          // print the time interval
   void   Report( int lvl, const string& formatString ); // conditionally prints the time interval

   const CTimer& operator=(const CTimer& src);  // Assignment
   const CTimer& operator-(const CTimer& src);  // subtraction

protected:
   void Init( bool fStart );
   void Copy( const CTimer& src );
   void Configure();

private:
   static long long  m_frequency;   // ticks per second
   long long m_start;               // timer start time.  positive if counting, negative if stopped
   };


inline void CTimer::Init( bool fStart )
   {
   if ( m_frequency == 0LL )
      {
      Configure();
      }

   m_start = 0LL;
   if ( fStart )
      {
      Start();
      }
   }

inline CTimer::CTimer( const CTimer& src )
   {
   Copy( src );
   }

inline void CTimer::Copy( const CTimer& src )
   {
   if ( this == &src )
      {
      return;                       // do not copy onto self
      }
   m_start = src.m_start;
   }

inline void CTimer::Start( bool fReset )
   {
   long long t;
   QueryPerformanceCounter((LARGE_INTEGER*)&t );
   if ( !fReset && (m_start < 0 ) )    // counter stopped w/ time on it
      {
      m_start += t;
      }
   else
      {
      m_start = t;
      }
   }

inline void CTimer::Stop()
   {
   if ( m_start <= 0 )     // already stopped
      {
      return;
      }
   long long t;
   QueryPerformanceCounter((LARGE_INTEGER*)&t );
   m_start -= t;
   }

inline double CTimer::Elapsed()
   {
   CTimer t(*this);
   t.Stop();
   double retVal = ((double)(-t.m_start))/(double)m_frequency; 
   return( retVal );
   }

inline double CTimer::Elapsedms()
   {
   CTimer t(*this);
   t.Stop();
   double retVal = ((-t.m_start)*1000.0)/(double)m_frequency; 
   return( retVal );
   }

inline double CTimer::Elapsedus()
   {
   CTimer t(*this);
   t.Stop();
   double retVal = ((-t.m_start)*1000000.0)/(double)m_frequency; 
   return( retVal );
   }

#endif   // Utils_h

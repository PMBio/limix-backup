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
 *         File Name:   Utils.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements a number of utility routines for console applications
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "Utils.h"

namespace plink {
	/*
	 * Globals we define
	 */
	int fVerbose = 0;             // global flag for verbose level output
	int fReportTimes = 0;         // global flag holds bit values to mask which times will be reported.

	static int cErrorsMax = 25;
	static int cErrors = 0;
	static int cWarns = 0;

	long long CTimer::m_frequency = 0ll;

	/*
	 * The code
	 */
	void Fatal(const char *szFmt, ...)
	{
		va_list va;

		fprintf(stderr, "\nFatal Error : ");

		va_start(va, szFmt);
		vfprintf(stderr, szFmt, va);
		va_end(va);

		fprintf(stderr, "\n");
		fflush(stderr);
		exit(1);
	}


	void Error(const char *szFmt, ...)
	{
		va_list va;

		++cErrors;
		fprintf(stderr, "\nError : ");

		va_start(va, szFmt);
		vfprintf(stderr, szFmt, va);
		va_end(va);
		if (cErrors >= cErrorsMax)
		{
			Fatal("Maximum error count exceeded. %d errors", cErrors);
		}
	}


	void FatalIfPreviousErrors()
	{
		if (cErrors)
		{
			Fatal("Exit with previous errors.");
		}
	}


	void Warn(const char *szFmt, ...)
	{
		va_list va;

		fprintf(stderr, "\nWarning : ");

		va_start(va, szFmt);
		vfprintf(stderr, szFmt, va);
		va_end(va);
		fflush(stderr);
	}


	void Verbose(const char *szFmt, ...)
	{
		if (!fVerbose)
		{
			return;
		}

		va_list va;

		fprintf(stderr, "\n");

		va_start(va, szFmt);
		vfprintf(stderr, szFmt, va);
		va_end(va);
	}

	void Progress(const char *szFmt, ...)
	{
		va_list va;

		va_start(va, szFmt);
		vfprintf(stderr, szFmt, va);
		va_end(va);
		fflush(stderr);
	}

	void ProgressNL(const char *szFmt, ...)
	{
		va_list va;

		fprintf(stderr, "\n");

		va_start(va, szFmt);
		vfprintf(stderr, szFmt, va);
		va_end(va);
		fflush(stderr);
	}


#if defined ( _MSC_VER )       // using Visual C compiler
	void NTLastError(void)
	{
		LPVOID lpMsgBuf;
		DWORD err = GetLastError();
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			err,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
			(LPTSTR)&lpMsgBuf,
			0,
			NULL);
		fprintf(stderr, "\nNTError : 0x%08X : %s", err, lpMsgBuf);
		fflush(stderr);
		LocalFree(lpMsgBuf);
		exit(1);
	}

#else    //#if !defined( _MSC_VER )

	void NTLastError(void)
	{
		Fatal("NTLastError stub for fatal error");
	}

	unsigned int _set_output_format(unsigned int format)
	{
		// stub a _set_output_format() routine for systems that don't have it
		static unsigned int old_format = 0;
		unsigned int rc = old_format;
		old_format = format;
		return(rc);
	}

	int GetComputerName(char *pchName, unsigned long* pcchMax)
	{
		static char szName[] = "ThisComputer";
		unsigned long l = *pcchMax;
		*pcchMax = sizeof(szName);
		if (l < sizeof(szName))
		{
			return(0);
		}
		strcpy(pchName, szName);
		return(1);
	}

#endif

	/*
	 *  File Handling helper routines
	 */
#if defined ( _MSC_VER )       // using Visual C compiler
#include <direct.h>
#endif

	/*
	 *  Check for existence of a file.
	 *  Return 1 (true) if file exists
	 *  return 0 (false) if no file or it is a directory
	 */
	static int FCheckFile(const std::string& name, int mask)
	{
		struct stat status;
		int rc = stat(name.c_str(), &status);
		if (rc == 0)
		{
			// status filled out successfully
			if ((status.st_mode & mask) == mask)
			{
				// file exists and mode matches mask requirements
				return(true);
			}
		}
		else if (rc == EINVAL)
		{
			Fatal("Failure in FCheckFile( %s, 0x%04X )"
				"\n  CRT Error: %d : %s", name.c_str(), mask, errno, strerror(errno));
		}
		return(false);
	}

	bool FIsFilenameWritable(const std::string& filename)
	{
		bool result = false;
		if (filename.size() == 0)
		{
			Fatal("FIsFileWritable( \"\" ) - No filename to be validated as Writable.");
		}

		struct stat status;
		int rc = stat(filename.c_str(), &status);
		if (rc == EINVAL)
		{
			Fatal("Failure in FIsFileWritable( %s )"
				"\n  CRT Error: %d : %s", filename.c_str(), errno, strerror(errno));
		}
		else if (rc == 0)
		{
			// the name in the file namespace is used!
			//  File or directory exists.  Is it a writeable file?
			if ((status.st_mode & (S_IFREG | S_IWRITE)) == (S_IFREG | S_IWRITE))
			{
				result = true;
			}
		}
		else
		{
			// stat says filename does not exist in the file namespace.
			//  Can we create it write to it?
			FILE *pFile = fopen(filename.c_str(), "wt");          // append ascii
			if (pFile)
			{
				fclose(pFile);
				remove(filename.c_str());
				result = true;
			}
		}
		return(result);
	}

#if 0
	int FWriteableFile(const std::string& name)
	{
		return(FCheckFile(name, S_IFREG | S_IWRITE));
	}

	int FWriteableDir(const std::string& name)
	{
		return(FCheckFile(name, S_IFDIR | S_IWRITE));
	}
#endif

	int FFileExists(const std::string& name)
	{
		return(FCheckFile(name, S_IFREG));
	}

	int FDirExists(const std::string& name)
	{
		return(FCheckFile(name, S_IFDIR));
	}

	void MakeDirectory(const std::string& dirname)
	{
		if (_access(dirname.c_str(), 0) == 0)
		{
			// The name exists in the filesystem
			struct stat status;
			stat(dirname.c_str(), &status);
			if (!(status.st_mode & S_IFDIR))
			{
				Fatal("File already exists.  Cannot create directory: [%s]", dirname.c_str());
			}
		}
		else
		{
			int rc = _mkdir((char *)dirname.c_str());
			if (rc == -1)
			{
				Fatal("Unable to create directory: [%s]"
					"\n  CRT Error: %d : %s", dirname.c_str(), errno, strerror(errno));
			}
		}
	}

	std::string FullPath(const std::string& filename)
	{
#if defined ( _MSC_VER )       // using Visual C compiler
		// Use Windows _fullpath() to expand the name
		char szFullPath[_MAX_PATH];
		_fullpath(szFullPath, filename.c_str(), sizeof(szFullPath));
		return(szFullPath);
#else
		// Use Linux realpath() to expand the name
		char szRealPath[PATH_MAX];
		char *szT = realpath(filename.c_str(), szRealPath);
		if (szT == nullptr)
		{
			strcpy(szRealPath, filename.c_str());

			// TODO: fix this for sure
			//Fatal( "Unable to resolve realpath() for filename: [%s]"
			//     "\n  %s", filename.c_str(), strerror( errno ) );

		}
		return(szRealPath);
#endif
	}

	std::string MakePath(const std::string& drive, const std::string& dir, const std::string& fname, const std::string& ext)
	{
#if defined ( _MSC_VER )         // using Visual C compiler
		char szFullPath[_MAX_PATH];
		_makepath(szFullPath, drive.c_str(), dir.c_str(), fname.c_str(), ext.c_str());
		return(szFullPath);
#else
		std::string t;
		if (drive.size() > 0)
		{
			if (drive[0] != '/')
			{
				t = "/";
			}
			t += drive;
		}
		if (dir.size() > 0)
		{
			if ((t.size() > 0) && (dir[0] != '/'))
			{
				t += "/";
			}
			t += dir;
		}
		if (fname.size() > 0)
		{
			if ((t.size() > 0) && (fname[0] != '/'))
			{
				t += "/";
			}
			t += fname;
		}
		if (ext.size() > 0)
		{
			t += "." + ext;
		}
		return(t);
#endif
	}

	void SplitPath(const std::string& filename, std::string& drive, std::string& dir, std::string& fname, std::string& ext)
	{
#if defined ( _MSC_VER )         // using Visual C compiler
		char szDrive[_MAX_DRIVE];
		char szDir[_MAX_DIR];
		char szFname[_MAX_FNAME];
		char szExt[_MAX_EXT];

		_splitpath(filename.c_str(), szDrive, szDir, szFname, szExt);
		drive = szDrive;
		dir = szDir;
		fname = szFname;
		ext = szExt;

#else
		// nothing like this on Linux. :-(
		//char *szDrive;
		char *szDir;
		char *szFname;
		char *szExt;

		char *szT = strdup(filename.c_str());
		char *pchSlash = strrchr(szT, '/');
		if (pchSlash == nullptr)
		{
			// it is all the filename
			szFname = szT;
		}
		else
		{
			// we have a path too
			*pchSlash = '\0';
			szDir = szT;
			szFname = pchSlash + 1;
		}
		char *pchDot = strrchr(szFname, '.');
		if (pchDot != nullptr)
		{
			*pchDot = '\0';
			szExt = pchDot + 1;
		}

		drive = "";
		dir = szDir;
		fname = szFname;
		ext = szExt;
		free(szT);
#endif
	}

	bool FIsCsvFile(const std::string& fname)
	{
		static char szCsvFileType[] = ".csv";
		size_t cchFileType = strlen(szCsvFileType);

		size_t cch = fname.size();
		if (cch > cchFileType)
		{
			const char *pch = fname.c_str() + fname.size() - cchFileType;
			if (_strcmpi(pch, szCsvFileType) == 0)
			{
				return(true);
			}
		}
		return(false);
	}


	/*
	 *  Higher resolution timing routines
	 */
	void CTimer::Configure()
	{
		QueryPerformanceFrequency((LARGE_INTEGER*)(&m_frequency));
		if (!m_frequency)
		{
			m_frequency = 1;
		}
		// This is where you'd remove overhead of start/stop routines, but
		//  it is too jittery
	}

	std::string CTimer::ToString(const std::string& formatString)
	{
		char rgch[256];

		sprintf(rgch, formatString.c_str(), ToString().c_str());
		std::string val(rgch);
		return(val);
	}

	std::string CTimer::ToString()
	{
		char rgch[64];

		CTimer timer(*this);
		timer.Stop();
		long long usecT = (-timer.m_start * 1000000LL) / m_frequency;
		int  usec = (int)(usecT % 1000LL);
		int  msec = (int)((usecT / 1000LL) % 1000LL);
		int  seconds = (int)((usecT / (1000LL * 1000LL)) % 60LL);
		int  minutes = (int)((usecT / (1000LL * 1000LL * 60LL)) % 60LL);
		int  hours = (int)((usecT / (1000LL * 1000LL * 60LL * 60LL)));
		if (hours)
		{
			sprintf(rgch, "%2d:%02d hrs", hours, minutes);
		}
		else if (minutes)
		{
			sprintf(rgch, "%2d:%02d mins", minutes, seconds);
		}
		else if (seconds)
		{
			sprintf(rgch, "%2d.%03d sec", seconds, msec);
		}
		else if (msec)
		{
			sprintf(rgch, "%3d.%03d ms", msec, usec);
		}
		else
		{
			sprintf(rgch, "%3d us", usec);
		}
		std::string s(rgch);
		return(s);
	}

	void CTimer::Report(const std::string& formatString)
	{
		ProgressNL(formatString.c_str(), ToString().c_str());
	}

	void CTimer::Report(int rptLevel, const std::string& formatString)
	{
		if ((fReportTimes & rptLevel) || (rptLevel == -1))
		{
			ProgressNL(formatString.c_str(), ToString().c_str());
		}
	}

	/*
	 * Other misc helper routines
	 */


	/*
	 *  Debug Dump routines
	 */
	void BinaryWrite(std::string fname, void* pv, size_t cb)
	{
		FILE *ofile = fopen(fname.c_str(), "wb");
		if (ofile == nullptr)
		{
			Fatal("Unable to open output file [%s]", fname.c_str());
		}
		if (pv)
		{
			fwrite(pv, 1, cb, ofile);
		}
		fclose(ofile);
	}

	static const char *openModeWrite = "w";
	static const char *openModeWriteAppend = "a";

#if 0
	void DumpLmmGwasInputs(StudyData *pStudyData, EigenSym *pEigenSym, ResultsLMM *pResultsLmm, bool fOpenAppend)
	{
		if (!fWriteLogFile)
		{
			return;
		}

		string logFile = MakePath("", logDir, "FastLmm.GwasInputs", "Log");
		const char *openMode = fOpenAppend ? openModeWriteAppend : openModeWrite;
		FILE *pf = fopen(logFile.c_str(), openMode);
		if (pf == nullptr)
		{
			Fatal("Unable to open log file: [%s]"
				"\n  CRT Error [%d]: %s", logFile.c_str(), errno, strerror(errno));
		}

		string commentString = "(DumpGwasInputs)";
		pStudyData->Dump(pf, commentString);
		pEigenSym->Dump(pf, commentString);
		// pResultsLmm->Dump( pf, commentString );
		fclose(pf);
	}

	void DumpStudyData(StudyData *pStudyData, const std::string& comment, bool fOpenAppend)
	{
		if (!fWriteLogFile)
		{
			return;
		}

		std::string logFile = MakePath("", logDir, "FastLmm.StudyData", "Log");
		const char *openMode = fOpenAppend ? openModeWriteAppend : openModeWrite;
		FILE *pf = fopen(logFile.c_str(), openMode);
		if (pf == nullptr)
		{
			Fatal("Unable to open log file: [%s]"
				"\n  CRT Error [%d]: %s", logFile.c_str(), errno, strerror(errno));
		}

		pStudyData->Dump(pf, comment);
		fclose(pf);
	}

	void DumpEigenSymState(EigenSym *pEigenSym, const std::string& comment, bool fOpenAppend)
	{
		if (!fWriteLogFile)
		{
			return;
		}

		std::string logFile = MakePath("", logDir, "FastLmm.EigenState", "Log");
		const char *openMode = fOpenAppend ? openModeWriteAppend : openModeWrite;
		FILE *pf = fopen(logFile.c_str(), openMode);
		if (pf == nullptr)
		{
			Fatal("Unable to open log file: [%s]"
				"\n  CRT Error [%d]: %s", logFile.c_str(), errno, strerror(errno));
		}

		pEigenSym->Dump(pf, comment);
		fclose(pf);
	}
#endif

	void DumpKernelArray(limix::mfloat_t *pReal, size_t cRows, size_t cColumns, const std::string& comment, bool fOpenAppend)
	{
		if (!fWriteLogFile)
		{
			return;
		}

		std::string logFile = MakePath("", logDir, "FastLmm.KernelArray", "Log");
		const char *openMode = fOpenAppend ? openModeWriteAppend : openModeWrite;
		FILE *pf = fopen(logFile.c_str(), openMode);
		if (pf == nullptr)
		{
			Fatal("Unable to open log file: [%s]"
				"\n  CRT Error [%d]: %s", logFile.c_str(), errno, strerror(errno));
		}

		fprintf(pf, "\nKernelArray: [%s]", comment.c_str());
		DumpRealArray(pf, 2, pReal, "Kernel", cRows, cColumns);
		fclose(pf);
	}

	void DumpStringVector(FILE* pf, int indent, const std::vector<std::string>& v, const std::string& name)
	{
#if defined( _MSC_VER )    // Windows/VC uses a %Iu specifier for size_t
		const char *szFmt1 = "\n%*svector<string> %s: [%Iu]";
		const char *szFmt2 = "\n%*s[%Iu]: %s";
#else                      // Linux/g++ uses a %zu specifier for size_t
		const char *szFmt1 = "\n%*svector<string> %s: [%zu]";
		const char *szFmt2 = "\n%*s[%zu]: %s";
#endif

		size_t cStrings = v.size();
		fprintf(pf, szFmt1, indent, "", name.c_str(), cStrings);
		for (size_t iString = 0; iString < cStrings; ++iString)
		{
			fprintf(pf, szFmt2, indent + 2, "", iString, v[iString].c_str());
		}
	}

	void DumpSnpInfoVector(FILE *pf, int indent, const std::vector<SnpInfo>& v, const std::string& name)
	{
#if defined( _MSC_VER )    // Windows/VC uses a %Iu specifier for size_t
		const char *szFmt1 = "\n%*svector<SnpInfo> %s: [%Iu]";
		const char *szFmt2 = "\n%*s[%Iu]: idSnp: %s";
#else                      // Linux/g++ uses a %zu specifier for size_t
		const char *szFmt1 = "\n%*svector<SnpInfo> %s: [%zu]";
		const char *szFmt2 = "\n%*s[%zu]: idSnp: %s";
#endif

		size_t c = v.size();
		fprintf(pf, szFmt1, indent, "", name.c_str(), c);
		for (size_t iSnpInfo = 0; iSnpInfo < c; ++iSnpInfo)
		{
			fprintf(pf, szFmt2, indent + 2, "", iSnpInfo, v[iSnpInfo].idSnp.c_str());
			fprintf(pf, "\n%*sChromosome: %d [%s]", indent + 4, "", v[iSnpInfo].iChromosome, v[iSnpInfo].idChromosome.c_str());
			fprintf(pf, "\n%*sMorgans: %e", indent + 4, "", v[iSnpInfo].geneticDistance);
			fprintf(pf, "\n%*sPosition: %d", indent + 4, "", v[iSnpInfo].basepairPosition);
			fprintf(pf, "\n%*sAlleles: %c %c", indent + 4, "", v[iSnpInfo].minorAllele, v[iSnpInfo].majorAllele);
		}
	}

	void DumpRealArray(FILE *pf, int indent, limix::mfloat_t *p, const std::string& name, size_t cRows, size_t cColumns)
	{
#if defined( _MSC_VER )    // Windows/VC uses a %Iu specifier for size_t
		const char *szFmt1 = "\n%*sreal* %s: [cRows: %Iu cColumns: %Iu]";
		const char *szFmt2 = "\n%*s[%Iu]:";
#else                      // Linux/g++ uses a %zu specifier for size_t
		const char *szFmt1 = "\n%*sreal* %s: [cRows: %zu cColumns: %zu]";
		const char *szFmt2 = "\n%*s[%zu]:";
#endif

		fprintf(pf, szFmt1, indent, "", name.c_str(), cRows, cColumns);
		if (p == nullptr)
		{
			fprintf(pf, "  ** nullptr **");
		}
		else
		{
			for (size_t iRow = 0; iRow < cRows; ++iRow)
			{
				fprintf(pf, szFmt2, indent + 2, "", iRow);
				for (size_t iColumn = 0; iColumn < cColumns; ++iColumn)
				{
					fprintf(pf, " %.17E", p[ColumnMajorIndex(cRows, cColumns, iRow, iColumn)]);
				}
			}
		}
	}

	void Dumpreals(FILE *pf, limix::mfloat_t *pdbl, size_t count)
	{
#if defined( _MSC_VER )    // Windows will zero fill w/ 0x%016p
		const char *szFmt1 = "\n 0x%016p: ";
#else                      // gcc/Linux doesn't like 0x%016p.  Use 0x%16p
		const char *szFmt1 = "\n 0x%16p: ";
#endif
		int i = 0;
		while (count--)
		{
			if (i == 0)
			{
				fprintf(pf, szFmt1, pdbl);
				i = 8;
			}
			fprintf(pf, " %8.3e", *pdbl);
			++pdbl;
			--i;
		}
		fprintf(pf, "\n");
	}

#if 0
	void DumpGroupSim(FILE *pf, int indent, GroupSim* p, const std::string& name)
	{
		fprintf(pf, "\n%*sGroupSim* %s: ", indent, "", name.c_str());
		if (p == nullptr)
		{
			fprintf(pf, "  ** nullptr **");
		}
		else
		{
			// Dump the rest of GroupSim Class members
			fprintf(pf, "\n%*s%s", indent + 2, "", " *** NYI DumpGroupSim ***");
		}
	}

#endif
}// end :plink
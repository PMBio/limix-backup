// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#if !defined( CPLINK_h )

#define CPLINK_h

#include "limix/types.h"
namespace plink {
#if defined( _MSC_VER )          // using Visual C compiler
#define  NOMINMAX
#include <windows.h>
#else // using gcc/g++ to compiler for Linux
#define  nullptr        NULL           // gcc does not yet support nullptr
	typedef  unsigned char  BYTE;          // common Windows types not found in gcc
	typedef  long long      __int64;
	typedef  long long      LARGE_INTEGER;
#endif
}// end :plink
#if defined( _MSC_VER )          // using Visual C compiler
#include <io.h>                  // for access()
#else
#include <unistd.h>              // for access() on POSIX/Linux
#endif

#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

#include <string>
#include <map>
#include <vector>
#include <list>
#include <limits>
#include <iostream>
#include <fstream>

#include "Utils.h"
#include "CLexer.h"
#include "CSnpInfo.h"
#include "CCovariatesData.h"
#include "CPlinkAlternatePhenotypeFile.h"
#include "CSnpFilterOptions.h"
#include "CPlinkFile.h"
namespace plink {
	const size_t NoMatch = -1;

	std::string missingPhenotypeString = "-9";  // default is -9
	limix::mfloat_t missingPhenotypeValue = -9.0;
	std::string missingGenotypeString = "-9";  // default is -9
	limix::mfloat_t missingGenotypeValue = -9.0;

	struct FamRecord              // or struct IndividualInfo... ?
	{
		// used by Fam and TFam
		std::string idFamily;           // can be alpha-numeric / or numeric... but no spaces
		std::string idIndividual;       // idIndiviual+idFamily must be unique!
		std::string idPaternal;         // must match another idIndividual ? (what is Family relationship?)
		std::string idMaternal;         // must match another idIndividual ? (what is Family relationship?)
		int    sex;                // 1=male, 2=female, other=unknown
		limix::mfloat_t   phenotype;          // -9=missing, 0=missing, 1=unaffected, 2=affected, or real (anything other than 0,1,2,-9)

		FamRecord() { sex = 0; phenotype = 0.0; }
	};

}//end :plink
#endif
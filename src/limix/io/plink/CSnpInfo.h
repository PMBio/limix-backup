#if !defined( CSnpInfo_h )
#define CSnpInfo_h
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

#include "Cplink.h"

namespace plink {

	struct SnpNucleotides         // what two nucleotides occur here
	{
		char alleles[2];

		SnpNucleotides() { Clear(); }
		void Clear() { alleles[0] = alleles[1] = '\0'; }
	};

	struct SnpProbabilities       // With what probabilities do each
	{                          //   major & minor nucleotide appear
		limix::mfloat_t probabilityOfHomozygousMinor;
		limix::mfloat_t probabilityOfHeterozygous;

		SnpProbabilities() { Clear(); }
		SnpProbabilities(limix::mfloat_t homozygousMinor, limix::mfloat_t heterozygous);
		void Clear() { probabilityOfHomozygousMinor = probabilityOfHeterozygous = 0.0; }
	};

	inline SnpProbabilities::SnpProbabilities(limix::mfloat_t homozygousMinor, limix::mfloat_t heterozygous)
	{
		probabilityOfHomozygousMinor = homozygousMinor;
		probabilityOfHeterozygous = heterozygous;
	}

	inline bool operator==(const SnpProbabilities& lhs, const SnpProbabilities& rhs)
	{
		if ((lhs.probabilityOfHomozygousMinor == rhs.probabilityOfHomozygousMinor)
			&& (lhs.probabilityOfHeterozygous == rhs.probabilityOfHeterozygous))
		{
			return(true);
		}
		return(false);
	};

	inline bool operator!=(const SnpProbabilities& lhs, const SnpProbabilities& rhs)
	{
		return(!(lhs == rhs));
	}

	struct SnpInfo                // information that describes a SNP
	{
		int    iChromosome;        // 1-22, X, Y, XY, MT, or 0 if unplaced
		std::string idChromosome;
		std::string idSnp;              // rs# or snp identifier
		limix::mfloat_t   geneticDistance;    // morgans
		int    basepairPosition;   // bp units
		char   majorAllele;        // A2 when reading/writing
		char   minorAllele;        // A1 when reading/writing

		SnpInfo() { iChromosome = basepairPosition = 0; geneticDistance = 0.0; majorAllele = minorAllele = '\0'; }
		void Clear() { iChromosome = basepairPosition = 0; geneticDistance = 0.0; majorAllele = minorAllele = '\0'; idChromosome.clear(); idSnp.clear(); }
	};
}//end :plink
#endif   // CSnpInfo_h

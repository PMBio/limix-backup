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
 * CPlinkLexer - {PLINK File Lexer Class}
 *
 *         File Name:   CPlinkLexer.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the CPlinkLexer class for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CPlinkLexer.h"

#if !defined( __GNUC__ )         // gcc doesn't yet support <regex>
#include <regex>

static std::tr1::regex rxInt( "[-+]?[0-9]+" );
static std::tr1::regex rxFp( "[-+]?([0-9]*\\.)?[0-9]+([eE][-+]?[0-9]+)?" );
static std::tr1::regex rxChromosome( "[0-9]+" );
#endif

namespace plink {
	int maxChromosomeValue = 26;

	CPlinkLexer::CPlinkLexer(const std::string& file) : CLexer(file)
	{
		// CLexer does all the initialization
	}

	CPlinkLexer::~CPlinkLexer()
	{
		// CLexer does all the cleanup right now
	}

	unsigned CPlinkLexer::NextToken(CToken& tok)
	{
	RetryAfterSkip:

		tok.type = tokError;
		tok.column = CurColumn();     // get the current position so we start collecting the token --  m_column;
		tok.line = CurLine();
		tok.offset = LexerFileOffset();
		tok.text.clear();

		switch (LookAheadChar())
		{
		case EOF:      // return EOF token
			tok.type = tokEOF;
			break;
		case '\t':     // White space - skip it all and return separator
		case ' ':
			SkipWhiteSpace();
			tok.type = tokSeparator;
			goto RetryAfterSkip;       // skip whitespace
			break;
		case '#':      // comment - skip to end of line
			SkipToEol();
			goto RetryAfterSkip;      // could be EOF
			break;
		case '\n':     // return a EOL token
			tok.type = tokEOL;
			AdvanceCh();
			break;
		default:       // see if it is a Symbol (ascii printable but NOT ' ')
			if ((LookAheadChar() < ' ') || (LookAheadChar() > '~'))
			{
				Fatal("Found invalid character in file [%s] near line: %d:%d", Filename().c_str(), tok.line, tok.column);
				break;
			}

			// accumulate ascii printable characters as a symbol
			while ((LookAheadChar() > ' ') && (LookAheadChar() <= '~'))
			{
				tok.text += (char)LookAheadChar();
				AdvanceCh();
			}
			tok.type = tokSymbol;
			break;
		}
		return(tok.type);
	}

	void CPlinkLexer::ExpectId(CToken& tok, std::string& output, const char *description)
	{
		if (tok.type != tokSymbol)     // get ID
		{
			Fatal("Expected %s in file [%s] near line %d:%d.  Found [%s]", description, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}

		output = tok.text;
		NextToken(tok);                // consumed the token so advance
	}

	void CPlinkLexer::ExpectInt(CToken& tok, int& output, const char *description)
	{
		if (tok.type != tokSymbol)     // get string
		{
			Fatal("Expected %s in file [%s] near line %d:%d.  Found [%s]", description, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}
#if !defined( __GNUC__ )         // gcc doesn't yet support <regex>
		//std::tr1::regex rxInt( "[-+]?[0-9]+" );       // move outside loop for perf
		if (!std::tr1::regex_match(tok.text, rxInt))
		{
			Fatal("Expected an integer %s in file [%s] near line %d:%d.  Found [%s]", description, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}
#endif
		output = atoi(tok.text.c_str());
		if (output == std::numeric_limits<int>::max())
		{
			Fatal("Integer overflow converting %s in file [%s] near line %d:d. Found [%s]", description, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}
		NextToken(tok);
	}

	void CPlinkLexer::ExpectReal(CToken& tok, limix::mfloat_t& output, const char *description)
	{
		if (tok.type != tokSymbol)     // get string representation
		{
			Fatal("Expected %s in file [%s] near line %d:%d.  Found [%s]", description, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}
#if !defined( __GNUC__ )         // gcc doesn't yet support <regex>
		//std::tr1::regex rxFp( "[-+]?([0-9]*\\.)?[0-9]+([eE][-+]?[0-9]+)?" );     // move outside loop for perf
		if (!std::tr1::regex_match(tok.text, rxFp))
		{
			Fatal("Expected a floating point number in file [%s] near line: %d:%d.  Found [%s]", Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}
#endif
		output = (limix::mfloat_t)atof(tok.text.c_str());
		NextToken(tok);
	}

	void CPlinkLexer::ExpectSex(CToken& tok, int& output)
	{
		if (tok.type != tokSymbol)     // get the fifth column - sex
		{
			Fatal("Expected a Sex indicator in file [%s] near line %d:%d.  Found [%s]", Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}
		if (tok.text.length() != 1)
		{
			Fatal("Expected Sex indicator in file [%s] near line %d:%d to be 1 character long.  Found %d characters [%s]",
				Filename().c_str(), tok.line, tok.column, tok.text.length(), tok.text.c_str());
		}
		char ch = tok.text[0];
		if (ch == '1') output = 1;
		else if (ch == '2') output = 2;
		else output = 0;
		NextToken(tok);
	}

	void CPlinkLexer::ExpectPhenotype(CToken& tok, limix::mfloat_t& output, int& outputType)
	{
		if (tok.type != tokSymbol)     //   this is trickier as it can be 0, 1, 2, -9, or an FP number
		{
			Fatal("Expected phenotype indicator in file [%s] near line %d:%d.  Found [%s]", Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}

		outputType = PHENOTYPE_AFFECTION;
		if (_strcmpi(tok.text.c_str(), "0") == 0) output = 0.0;
		else if (_strcmpi(tok.text.c_str(), "1") == 0) output = 1.0;
		else if (_strcmpi(tok.text.c_str(), missingPhenotypeString.c_str()) == 0) output = std::numeric_limits<limix::mfloat_t>::quiet_NaN();
		else
		{
			// must be an FP number
#if !defined( __GNUC__ )         // gcc doesn't yet support <regex>
			//std::tr1::regex rxFp( "[-+]?([0-9]*\\.)?[0-9]+([eE][-+]?[0-9]+)?" );     // move outside loop for perf
			if (!std::tr1::regex_match(tok.text, rxFp))
			{
				Fatal("Expected phenotype indicator in file [%s] near line %d:%d to be 0, 1, -9, or a floating point number.  Found [%s]",
					Filename().c_str(), tok.line, tok.column, tok.text.c_str());
			}
#endif
			output = (limix::mfloat_t)atof(tok.text.c_str());
			if (output == missingPhenotypeValue)
			{
				output = std::numeric_limits<limix::mfloat_t>::quiet_NaN();
			}
			outputType = PHENOTYPE_QUANTITATIVE;
		}
		NextToken(tok);
	}

	void CPlinkLexer::ExpectAllele(CToken& tok, char& output, const char *description)
	{
		if ((tok.type != tokSymbol) || (tok.text.length() != 1))
		{
			Fatal("Expected %s allele of genotype pair in file [%s] near line %d:%d.  Found %s", description, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}

		output = tok.text[0];
		NextToken(tok);                // consumed the token so advance
	}

	void CPlinkLexer::ExpectSnpNucleotides(CToken& tok, char& firstNucleotide, char& secondNucleotide)
	{
		size_t tokenLine = tok.line;
		size_t tokenColumn = tok.column;
		ExpectAllele(tok, firstNucleotide, "First");
		ExpectAllele(tok, secondNucleotide, "Second");
		if (((firstNucleotide == '0') && (secondNucleotide != '0'))
			|| ((firstNucleotide != '0') && (secondNucleotide == '0'))
			)
		{
			// bad genotype data.  Mixed a missing and a non-missing character
			// TODO:  PLink documentation calls out that mixing missing and
			//         not missing characters for the genotype is illegal
			//         and the user should resolve the data to provide a 
			//         homozygous genotype or set both alleles to missing.
			//
			//         I tend to prefer to not parse 'illegal' input as it 
			//         causes 'undefined' behavior to occur.  
			//         In most cases, the user should fix the input.
			//
			//   Fatal
			Warn("Inconsistent missing genotype information found in file [%s] near line %d:%d."
				"\n  found [%c] [%c]", Filename().c_str(), tokenLine, tokenColumn, firstNucleotide, secondNucleotide);
			Warn("    Genotype information treated as missing");
			firstNucleotide = '0';
			secondNucleotide = '0';
		}
		return;
	}

	void CPlinkLexer::ExpectSnpAlleles(CToken& tok, char& majorAllele, char& minorAllele)
	{
		size_t tokenLine = tok.line;
		size_t tokenColumn = tok.column;
		ExpectAllele(tok, minorAllele, "minor");
		ExpectAllele(tok, majorAllele, "major");
		if ((majorAllele == '0') || (minorAllele == '0'))
		{
			// bad allele data for SNP.  Must have allele information
			Fatal("Found missing SNP minor or major allele information in file [%s] near line %d:%d."
				"\n  found [%c] [%c]", Filename().c_str(), tokenLine, tokenColumn, minorAllele, majorAllele);
		}
		return;
	}

	void CPlinkLexer::ExpectProbability(CToken& tok, limix::mfloat_t& output, const char *description)
	{
		size_t tokenLine = tok.line;
		size_t tokenColumn = tok.column;
		ExpectReal(tok, output, description);
		if ((output != -9.0) && ((output < 0.0) || (output > 1.0)))
		{
			Fatal("%s SNP probability out of range in file [%s] near line %d:%d."
				"\n  Expected a number between 0.0 and 1.0, but found %.13e",
				description, Filename().c_str(), tokenLine, tokenColumn, output);
		}
	}

	void CPlinkLexer::ExpectSnpProbabilities(CToken& tok, SnpProbabilities& output)
	{
		size_t tokenLine = tok.line;
		size_t tokenColumn = tok.column;
		ExpectProbability(tok, output.probabilityOfHomozygousMinor, "Probability Of Homozygous Minor Genotype");
		ExpectProbability(tok, output.probabilityOfHeterozygous, "Probability Of Heterozygous Genotype");

		if (((output.probabilityOfHomozygousMinor == missingGenotypeValue) && (output.probabilityOfHeterozygous != missingGenotypeValue))
			|| ((output.probabilityOfHomozygousMinor != missingGenotypeValue) && (output.probabilityOfHeterozygous == missingGenotypeValue))
			)
		{
			// bad genotype data.  Mixed a missing and a non-missing character
			//
			Warn("Inconsistent missing genotype information found in file [%s] near line %d:%d."
				"\n  found [%f] [%f]", Filename().c_str(), tokenLine, tokenColumn, output.probabilityOfHomozygousMinor, output.probabilityOfHeterozygous);
			Warn("    Genotype information treated as missing");
			output.probabilityOfHomozygousMinor = -9.0;//TODO value hardcoded
			output.probabilityOfHeterozygous = -9.0;//TODO value hardcoded
		}
		else
		{
			limix::mfloat_t sum = output.probabilityOfHomozygousMinor + output.probabilityOfHeterozygous;
			if (sum > 1.0)
			{
				Fatal("Expected sum of SNP probabilities to be no less than 0.0 and no greater than 1.0."
					"\n  Found %f and %f in file [%s] near line %d:%d",
					output.probabilityOfHomozygousMinor,
					output.probabilityOfHeterozygous,
					Filename().c_str(),
					tokenLine,
					tokenColumn);
			}
		}
	}

	void CPlinkLexer::ExpectChromosome(CToken& tok, std::string& output, int& intOutput)
	{
		if (tok.type != tokSymbol)     //   this is trickier as it can be and integer 0-26 or 'X', 'Y', 'XY', or 'MT'
		{
			Fatal("Expected Chromosome in file [%s] near line %d:%d.  Found [%s]", Filename().c_str(), tok.line, tok.column, tok.text.c_str());
		}

		if (_strcmpi(tok.text.c_str(), "x") == 0) intOutput = 23;
		else if (_strcmpi(tok.text.c_str(), "y") == 0) intOutput = 24;
		else if (_strcmpi(tok.text.c_str(), "xy") == 0) intOutput = 25;
		else if (_strcmpi(tok.text.c_str(), "mt") == 0) intOutput = 26;
		else
		{
#if !defined( __GNUC__ )         // gcc doesn't yet support <regex>
			//std::tr1::regex rxChromosome( "[0-9]+" );        // move outside loop for perf
			if (!std::tr1::regex_match(tok.text, rxChromosome))
			{
				char szFmt[] = "Expected a Chromosome in the form of and integer or a string X, Y, XY, or MT in file [%s] near line %d:%d.  Found [%s]";
				Fatal(szFmt, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
			}
#endif
			int i = atoi(tok.text.c_str());
			if (i > maxChromosomeValue)
			{
				char szFmt[] = "Expected a Chromosome in the form of and integer from 0-%d in file [%s] near line %d:%d.  Found [%s]";
				Fatal(szFmt, maxChromosomeValue, Filename().c_str(), tok.line, tok.column, tok.text.c_str());
			}
			intOutput = i;
		}

		output = tok.text;
		NextToken(tok);
	}
}// end :plink
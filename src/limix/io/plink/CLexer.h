#if !defined( CLexer_h )
#define CLexer_h
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
 * CLexer - {Simple Scanner/Lexer Functionality}
 *
 *         File Name:   CLexer.h [Simple Lexer Class
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file declares a routines for a basic lexer/scanner
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Get our includes
 */
#include "Cplink.h"
/*
 * 'Publish' our defines
 */
namespace plink {

	typedef enum _TokType
	{
		tokError = 0
		, tokTab = '\t'
		, tokEOF = 256
		, tokEOL
		, tokSymbol
		, tokSeparator
	} TokType;

	class CToken
	{
	public:
		CToken() { type = tokError; offset = 0; line = column = 0; };

		TokType     type;       // stores Token Type
		long long   offset;     // track file position of token
		size_t      line;
		size_t      column;
		std::string      text;
	};

	class CLexer
	{
	public:
		CLexer(const std::string& filename);
		~CLexer();

		void        AdvanceCh();
		void        SkipToEol();
		void        SkipWhiteSpace();
		int         LookAheadChar() { return(m_chLook); }
		size_t      CurColumn() { return(m_column); }
		size_t      CurLine()   { return(m_line); }
		std::string&     Filename()  { return(m_fileName); }
		long long   LexerFileOffset() { return(_ftelli64(m_pFile) - 1); }   // already read the first character

	private:
		std::string   m_fileName;
		FILE     *m_pFile;

		// track file position
		unsigned long m_line;
		unsigned long m_column;       // expanded for tabs
		int      m_cchInLine;         // count of characters in the line (

		int      m_chLook;            // look ahead
		std::string   m_text;
	};
}// end: plink
#endif      // CLexer

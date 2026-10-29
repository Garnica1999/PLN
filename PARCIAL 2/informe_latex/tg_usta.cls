
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{tg_usta}[2008/10/01 v1.0f]

\RequirePackage[spanish]{babel}
\RequirePackage[utf8]{inputenc}                  % fuente utilizada para incorporar simbolos acentuados y otros s�mbolos
\RequirePackage{amsmath,amsthm,amssymb,latexsym}    % herramientas para manejo matem�tico y s�mbolos matem�ticos especiales
\RequirePackage{verbatim}                           % herramienta para incluir texto en fuente mono-espaciada y para comentarios
\RequirePackage{enumerate}                          % herramienta para el manejo de listas enumeradas y encajadas
\RequirePackage{array}                              % utilizado para darle formato al encabezado de las tablas            
\RequirePackage{ifpdf}
\ifpdf % herramientas para inclusi�n de gr�ficos y el manejo del color
    \RequirePackage[pdftex]{graphicx,color}
\else
    \RequirePackage[dvips]{graphicx,color}%
\fi

\def\BibTeX{B{\footnotesize IB}\TeX}                % para agregar la palabra BibTeX en estilo LaTeX
\def\MiKTeX{M{\footnotesize iK}\TeX}                % para agregar la palabra MikTeX en estilo LaTeX
\def\emphred#1{{\color{red}#1}}                     % para resaltar texto dudoso a corregir

\definecolor{lightgray}{gray}{0.5}   

\def\Hy@colorlink{}
\def\Hy@endcolorlink{}

\def\@maintitle{}
\def\@secondtitle{}
\def\@shorttitle{}
\def\@firstpage{1}
\def\@lastpage{\label{@@lastpage}}
\def\@volume{0}
\def\@number{0}
\def\@month{0}
\def\@year{0000}

\newcounter{@author}
\setcounter{@author}{1}
\newcounter{@headauthor}
\setcounter{@headauthor}{0}

\def\@languagespanish{spanish}
\def\@languageenglish{english}
\def\@mathpazo{}
\def\@cover{}
\def\@frontcover{}

\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\input{size1\@ptsize.clo}
    \setlength\paperheight{11in}%
    \setlength\paperwidth{8.5in}%
    \setlength\textheight{195mm}
    \setlength\textwidth{125mm}
    \setlength\topmargin{0cm}   
    \setlength\parskip{2mm}      
    \setlength\marginparwidth{2mm}               
    \setlength\headsep{6mm}                       
    %\setlength\oddsidemargin{15mm}                
    %\setlength\evensidemargin{-4mm} 
    \setlength\voffset{-5.4mm}
    \setlength\parindent{0cm}} 
\DeclareOption{cartapaper}
   {\input{size1\@ptsize.clo}
    \setlength\paperheight{11in}%
    \setlength\paperwidth{8.5in}%
    \setlength\textheight{195mm}
    \setlength\textwidth{125mm}
    \setlength\topmargin{0cm}   
    \setlength\parskip{2mm}      
    \setlength\marginparwidth{2mm}               
    \setlength\headsep{6mm}                       
    %\setlength\oddsidemargin{15mm}                
    %\setlength\evensidemargin{-4mm} 
    \setlength\voffset{-5.4mm}
    \setlength\parindent{0cm}} 
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}

\DeclareOption{spanish}{
    \def\@language{spanish}
    \def\mainabstractname{Resumen}
    \def\secondaryabstractname{Abstract}
    \def\mainkeywordsname{Palabras clave}
    \def\secondarykeywordsname{Keywords}
}

\DeclareOption{english}{
    \def\@language{english}
    \def\mainabstractname{Abstract}
    \def\secondaryabstractname{Resumen}
    \def\mainkeywordsname{Keywords}
    \def\secondarykeywordsname{Palabras clave}
}

\DeclareOption{mathpazo}{
    \def\@mathpazo{mathpazo}
}

\DeclareOption{cover}{
    \def\@cover{cover}
}

\DeclareOption{frontcover}{
    \def\@frontcover{frontcover}
}

\ExecuteOptions{10pt,letterpaper,twoside,onecolumn,final,spanish}
\ProcessOptions   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifx\@empty\@cover
    \ifx\@empty\@frontcover
        \AtBeginDocument{\pagestyle{mypagehead}\thispagestyle{plain}\def\@roman#1{\romannumeral #1}}
    \else\definecolor{backgroundcolor}{rgb}{0,0.49,0.76}\pagecolor{backgroundcolor}\fi
\else 
\AtBeginDocument{\includegraphics[scale=0.4]{logo_javeriana.png}\hfill
                \renewcommand{\tabcolsep}{0pt}
                  \begin{tabular}[b]{c} Maestría en Inteligencia Artificial \\\\[-2ex]\end{tabular}\\[-0.7em]\rule{\textwidth}{1.5pt}\\[-3.5ex]
                 \pagestyle{empty}\def\@roman#1{\romannumeral #1}}
\fi

\AtEndDocument{\@lastpage}

\newcommand\frontcover{
 \textbf{       \@@Month\ \@year, Vol. \@volume, No. \@number}
}

\ifx\@language\@languagespanish 
    \RequirePackage[spanish]{babel}         % para definir el idioma en que se van a escribir los art�culos
    \addto\layoutspanish{\def\@seccntformat#1{\csname the#1\endcsname.\space}}
    \decimalpoint 
    \deactivatetilden        
    \addto\captionsspanish{
            \def\tablename{Tabla}
    }
\else
    \RequirePackage[english]{babel}         % para definir el idioma en que se van a escribir los art�culos
\fi   

\ifx\@empty\@mathpazo\else\RequirePackage{mathpazo}\fi  

\def\labelpaper{\@ifnextchar[\@valuePaper{}} 
\def\@valuePaper[#1]{\setkeys{valuespaper}{#1}}

\define@key{valuespaper}{firstpage}{\def\@firstpage{#1}\setcounter{page}{\@firstpage}}
\define@key{valuespaper}{volume}{\def\@volume{#1}}
\define@key{valuespaper}{number}{\def\@number{#1}}
\define@key{valuespaper}{month}{\def\@month{#1}}
\define@key{valuespaper}{year}{\def\@year{#1}}
\def\@@month{\ifcase\@month\or enero\or febrero\or marzo\or abril\or mayo\or junio\or julio\or agosto\or septiembre\or octubre\or noviembre\or diciembre\fi}
\def\@@Month{\ifcase\@month\or Enero\or Febrero\or Marzo\or Abril\or Mayo\or Junio\or Julio\or Agosto\or Septiembre\or Octubre\or Noviembre\or Diciembre\fi}

\def\title{\@ifnextchar[\@valueTitle{}}%
\def\@valueTitle[#1]{%
    \setkeys{valuestitle}{#1}
    \begin{center}
    \includegraphics[scale=0.07]{logo_javeriana.png}\hfill
    \begin{footnotesize}
	\renewcommand{\tabcolsep}{0pt}
	\begin{tabular}[b]{c} {\bf Facultad de Ingenier\'ia}\\Maestr\'ia en Inteligencia Artificial \\ Parcial 2 - NLP\\ \\[-3ex]\end{tabular}
    \end{footnotesize}  
    \end{center}
    \noindent\\[-2.5em]\rule{\textwidth}{1.5pt}\\[-3.5ex]
    \begin{center}\Large\bfseries \@maintitle \end{center}
    \begin{center}\normalsize\bfseries \@secondtitle \end{center}
    \ifx\@empty\@shorttitle
    \gdef\rightmark{\@maintitle}
    \else
    \gdef\rightmark{\@shorttitle}
    \fi
}
\define@key{valuestitle}{maintitle}{\def\@maintitle{#1}}
\define@key{valuestitle}{secondtitle}{\def\@secondtitle{#1}}
\define@key{valuestitle}{shorttitle}{\def\@shorttitle{#1}}

\newenvironment{authors}[1][]{%
    \xdef\@thereIsShortAuthor{#1} 
    \setcounter{footnote}{0}\def\thefootnote{\@alph\c@footnote}
}%
{%
\renewcommand{\tabcolsep}{0pt}
\begin{tabular}{cc}
\parbox{0.5\textwidth}{\quad} & \parbox{0.5\textwidth}{\quad} \\[-2.1ex]
\@namecolumn\\[4mm]
\end{tabular}
\par\noindent\rule{\textwidth}{1.5pt}\\[-2.5ex]
\ifx\@empty\@thereIsShortAuthor\else\xdef\leftmark{\@thereIsShortAuthor}\fi
\setcounter{footnote}{0}\def\thefootnote{\@arabic\c@footnote}}
    
\def\author{\@ifnextchar[\@valueAuthor{}}
\def\@valueAuthor[#1]{%
    \setkeys{valuesauthor}{#1}\@name%
}

\define@key{valuesauthor}{firstname}{
\def\@firstname{#1}
\ifcase\value{@author}%
\or \gdef\@firstnameA{#1}
\or \gdef\@firstnameB{#1}
\or \gdef\@firstnameC{#1}
\or \gdef\@firstnameD{#1}
\or \gdef\@firstnameE{#1}
\or \gdef\@firstnameF{#1}
\or \gdef\@firstnameG{#1}
\or \gdef\@firstnameH{#1}
\or \gdef\@firstnameI{#1}
\or \gdef\@firstnameJ{#1}
\fi
}  

\define@key{valuesauthor}{surname}{
\def\@surname{#1}
\ifcase\value{@author}%
\or \gdef\@surnameA{#1}
\or \gdef\@surnameB{#1}
\or \gdef\@surnameC{#1}
\or \gdef\@surnameD{#1}
\or \gdef\@surnameE{#1}
\or \gdef\@surnameF{#1}
\or \gdef\@surnameG{#1}
\or \gdef\@surnameH{#1}
\or \gdef\@surnameI{#1}
\or \gdef\@surnameJ{#1}
\fi                             
}


\define@key{valuesauthor}{affiliation}{\def\@affiliation{#1}}

\define@key{valuesauthor}{email}{
\ifcase\value{@author}%
\or \gdef\@emailA{#1}
\or \gdef\@emailB{#1}
\or \gdef\@emailC{#1}
\or \gdef\@emailD{#1}
\or \gdef\@emailE{#1}
\or \gdef\@emailF{#1}
\or \gdef\@emailG{#1}
\or \gdef\@emailH{#1}
\or \gdef\@emailI{#1}
\or \gdef\@emailJ{#1}
\fi
}

\def\mul@#1#2{\multicolumn{#1}{c}{#2}}

\def\@name{
    \ifcase\value{@author}%
    \or
        \def\@numcolA{2}
        \gdef\@namecolumn{\mul@{\@numcolA}{\parbox{0.5\textwidth}{\centering\@firstnameA\ \@surnameA\\[-.75mm]{\footnotesize\@emailA}}}}
    \or
        \def\@numcolA{1}
        \def\@numcolB{1}
        \g@addto@macro\@namecolumn{&\mul@{\@numcolB}{\parbox{0.5\textwidth}{\centering\@firstnameB\ \@surnameB\\[-.75mm]{\footnotesize\@emailB}}}}
    \or
        \def\@numcolC{2}
        \g@addto@macro\@namecolumn{\\[4mm]\mul@{\@numcolC}{\parbox{0.5\textwidth}{\centering\@firstnameC\ \@surnameC\\[-.75mm]{\footnotesize\@emailC}}}}
    \or
        \def\@numcolC{1}
        \def\@numcolD{1}
        \g@addto@macro\@namecolumn{&\mul@{\@numcolD}{\parbox{0.5\textwidth}{\centering\@firstnameD\ \@surnameD\\[-.75mm]{\footnotesize\@emailD}}}}
    \or
        \def\@numcolE{2}
        \g@addto@macro\@namecolumn{\\[4mm]\mul@{\@numcolE}{\parbox{0.5\textwidth}{\centering\@firstnameE\ \@surnameE\footnotemark\\[-.75mm]{\footnotesize\@emailE}}}}
    \or
        \def\@numcolE{1}
        \def\@numcolF{1}
        \g@addto@macro\@namecolumn{&\mul@{\@numcolF}{\parbox{0.5\textwidth}{\centering\@firstnameF\ \@surnameF\footnotemark\\[-.75mm]{\footnotesize\@emailF}}}}
    \or
        \def\@numcolG{2}
        \g@addto@macro\@namecolumn{\\[4mm]\mul@{\@numcolG}{\parbox{0.5\textwidth}{\centering\@firstnameG\ \@surnameG\footnotemark\\[-.75mm]{\footnotesize\@emailG}}}}
    \or
        \def\@numcolG{1}
        \def\@numcolH{1}
        \g@addto@macro\@namecolumn{&\mul@{\@numcolH}{\parbox{0.5\textwidth}{\centering\@firstnameH\ \@surnameH\footnotemark\\[-.75mm]{\footnotesize\@emailH}}}}
    \or
        \def\@numcolI{2}
        \g@addto@macro\@namecolumn{\\[4mm]\mul@{\@numcolI}{\parbox{0.5\textwidth}{\centering\@firstnameI\ \@surnameI\footnotemark\\[-.75mm]{\footnotesize\@emailI}}}}
    \or
        \def\@numcolI{1}
        \def\@numcolJ{1}
        \g@addto@macro\@namecolumn{&\mul@{\@numcolJ}{\parbox{0.5\textwidth}{\centering\@firstnameJ\ \@surnameJ\footnotemark\\[-.75mm]{\footnotesize\@emailJ}}}}
    \fi
   %\footnotetext[\value{@author}]{\@affiliation}
    \stepcounter{@author} 
    
    \ifnum\value{@headauthor} = 0%
        \xdef\@headnames{\@firstname\ \@surname}%
        \xdef\@headnamesaux{\@firstname\ \@surname}%
    \else%
        \xdef\@headnames{\@headnamesaux\ \& \@firstname\ \@surname}%
        \xdef\@headnamesaux{\@headnamesaux, \@firstname\ \@surname}%      
    \fi%    
    \stepcounter{@headauthor}% 
    \xdef\leftmark{\@headnames} 
}

\newenvironment{mainabstract}{%
        \def\abstractname{\mainabstractname}
        \def\keywordsname{\mainkeywordsname}
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
}{}

\newenvironment{secondaryabstract}{%
      \def\abstractname{\secondaryabstractname}
      \def\keywordsname{\secondarykeywordsname}
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
}{}

\def\keywords#1{\\[1.5ex]{\bfseries\textit\keywordsname:\;\unskip}#1.}


\newcommand{\references}[1]{%
    \bibliographystyle{agsm}%
    \bibliography{#1}%
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
%\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}


\def\ps@mypagehead{        
      \def\@oddfoot{\hfil\footnotesize\bf{\bf Facultad de Ingenier\'ia}\\ Parcial 2 - NLP\\}%
      \def\@evenfoot{\hfil\footnotesize\bf {\bf Facultad de Ingenier\'ia}\\ Parcial 2 - NLP\\}%
      \def\@evenhead{\bf\thepage\hfil{\footnotesize\leftmark}}%
      \def\@oddhead{\bf{\footnotesize\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection\quad
        \fi
        ##1}}{}}%
    \def\subsectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\@ne
          \thesubsection\quad
        \fi
        ##1}}
}

\setcounter{secnumdepth}{3}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\part{%
   \if@noskipsec \leavevmode \fi
   \par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \Large\bfseries \partname\nobreakspace\thepart
       \par\nobreak
     \fi
     \huge \bfseries #2%
     \markboth{}{}\par}%
    \nobreak
    \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \huge \bfseries #1\par}%
     \nobreak
     \vskip 3ex
     \@afterheading}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\Large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\large\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
%\newcommand\labelenumi{\theenumi.}
%\newcommand\labelenumii{(\theenumii)}
%\newcommand\labelenumiii{\theenumiii.}
%\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
%\newcommand\labelitemi{\textbullet}
%\newcommand\labelitemii{\normalfont\bfseries \textendash}
%\newcommand\labelitemiii{\textasteriskcentered}
%\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand\tableofcontents{%
    \section*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \@starttoc{toc}%
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
       \if@compatibility
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
      \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand\listoffigures{%
    \section*{\listfigurename}%
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}%
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \section*{\listtablename}%
      \@mkboth{%
          \MakeUppercase\listtablename}%
         {\MakeUppercase\listtablename}%
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\section*{\centerline\refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\section*{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 35\p@
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\refname{References}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
%\pagestyle{plain}
\pagenumbering{arabic}

\RequirePackage[abbr]{harvard}                      % utilizado para determinar el estilo de la bibliografia
\def\harvardurl#1{*#1}

\endinput
%%
%% End of file `article.cls'.

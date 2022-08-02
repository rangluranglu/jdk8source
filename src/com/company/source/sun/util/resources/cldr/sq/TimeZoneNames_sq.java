/*
 * Copyright (c) 2012, 2018, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/*
 * COPYRIGHT AND PERMISSION NOTICE
 *
 * Copyright (C) 1991-2012 Unicode, Inc. All rights reserved. Distributed under
 * the Terms of Use in http://www.unicode.org/copyright.html.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of the Unicode data files and any associated documentation (the "Data
 * Files") or Unicode software and any associated documentation (the
 * "Software") to deal in the Data Files or Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Data Files or Software, and
 * to permit persons to whom the Data Files or Software are furnished to do so,
 * provided that (a) the above copyright notice(s) and this permission notice
 * appear with all copies of the Data Files or Software, (b) both the above
 * copyright notice(s) and this permission notice appear in associated
 * documentation, and (c) there is clear notice in each modified Data File or
 * in the Software as well as in the documentation associated with the Data
 * File(s) or Software that the data or software has been modified.
 *
 * THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
 * KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
 * THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS
 * INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR
 * CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
 * DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 * TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THE DATA FILES OR SOFTWARE.
 *
 * Except as contained in this notice, the name of a copyright holder shall not
 * be used in advertising or otherwise to promote the sale, use or other
 * dealings in these Data Files or Software without prior written authorization
 * of the copyright holder.
 */

package sun.util.resources.cldr.sq;

import sun.util.resources.TimeZoneNamesBundle;

public class TimeZoneNames_sq extends TimeZoneNamesBundle {
    @Override
    protected final Object[][] getContents() {
        final String[] Moscow = new String[] {
               "Ora standarde e Mosk\u00ebs",
               "MST",
               "Ora verore e Mosk\u00ebs",
               "MST",
               "Ora e Mosk\u00ebs",
               "MT",
            };
        final String[] Europe_Central = new String[] {
               "Ora standarde qendrore evropiane",
               "CEST",
               "Ora verore qendrore evropiane",
               "CEST",
               "Ora qendrore evropiane",
               "CET",
            };
        final String[] Europe_Eastern = new String[] {
               "Ora standarde lindore evropiane",
               "EEST",
               "Ora verore lindore evropiane",
               "EEST",
               "Ora lindore evropiane",
               "EET",
            };
        final Object[][] data = new Object[][] {
            { "Europe/Paris", Europe_Central },
            { "Europe/Bucharest", Europe_Eastern },
            { "Europe/Samara", Moscow },
            { "Europe/Sofia", Europe_Eastern },
            { "Europe/Monaco", Europe_Central },
            { "Europe/Gibraltar", Europe_Central },
            { "Europe/Vienna", Europe_Central },
            { "Asia/Damascus", Europe_Eastern },
            { "Europe/Malta", Europe_Central },
            { "Europe/Madrid", Europe_Central },
            { "Europe/Minsk", Europe_Eastern },
            { "Europe/Vilnius", Europe_Eastern },
            { "Europe/Mariehamn", Europe_Eastern },
            { "Europe/Podgorica", Europe_Central },
            { "Asia/Nicosia", Europe_Eastern },
            { "Europe/Riga", Europe_Eastern },
            { "Europe/Luxembourg", Europe_Central },
            { "Europe/Zurich", Europe_Central },
            { "Asia/Amman", Europe_Eastern },
            { "Europe/Brussels", Europe_Central },
            { "Europe/Zaporozhye", Europe_Eastern },
            { "Africa/Tripoli", Europe_Eastern },
            { "Europe/Simferopol", Europe_Eastern },
            { "Europe/Oslo", Europe_Central },
            { "Europe/Rome", Europe_Central },
            { "Europe/Vatican", Europe_Central },
            { "Europe/Tirane", Europe_Central },
            { "Europe/Istanbul", Europe_Eastern },
            { "Europe/Copenhagen", Europe_Central },
            { "Europe/Helsinki", Europe_Eastern },
            { "Europe/Tallinn", Europe_Eastern },
            { "Europe/Amsterdam", Europe_Central },
            { "Europe/Athens", Europe_Eastern },
            { "Asia/Hebron", Europe_Eastern },
            { "Europe/Uzhgorod", Europe_Eastern },
            { "Europe/Stockholm", Europe_Central },
            { "Europe/Berlin", Europe_Central },
            { "Europe/Skopje", Europe_Central },
            { "Arctic/Longyearbyen", Europe_Central },
            { "Africa/Ceuta", Europe_Central },
            { "Europe/Andorra", Europe_Central },
            { "Europe/Chisinau", Europe_Eastern },
            { "Asia/Gaza", Europe_Eastern },
            { "Europe/Budapest", Europe_Central },
            { "Africa/Tunis", Europe_Central },
            { "Asia/Beirut", Europe_Eastern },
            { "Europe/San_Marino", Europe_Central },
            { "Europe/Vaduz", Europe_Central },
            { "Europe/Sarajevo", Europe_Central },
            { "Europe/Prague", Europe_Central },
            { "Europe/Bratislava", Europe_Central },
            { "Europe/Ljubljana", Europe_Central },
            { "Europe/Zagreb", Europe_Central },
            { "Africa/Algiers", Europe_Central },
            { "Europe/Warsaw", Europe_Central },
            { "Europe/Kiev", Europe_Eastern },
            { "Africa/Cairo", Europe_Eastern },
            { "Europe/Belgrade", Europe_Central },
            { "Europe/Moscow", Moscow },
        };
        return data;
    }
}

/*
 * Copyright (C) Foswiki Contributors 2010-2014
 * Author: Crawford Currie http://c-dot.co.uk
 * Javascript implementation of TablePlugin
 * Uses the jQuery tablesorter plugin
 */
(function($){var jqtp={tableframe:{'void':'none','above':'solid none none none','below':'none none solid none','lhs':'none none none solid','rhs':'none solid none none','hsides':'solid none solid none','vsides':'none solid none solid','box':'solid','border':'solid'},defaultOpts:{debug:false,widgets:['colorize']},move_table_data:function(elem){var $this=$(elem),params=$this.data(),$table=$(jqtp.next_table(elem));$this.removeClass("jqtp_process");if(!$table.length){return;}
$table.data('jqtp_process',params);$this.remove();},next_table:function(el){if(el===null){return null;}
do{while(el.nextSibling===null){if(el.parentNode===null){return null;}else{el=el.parentNode;}}
el=el.nextSibling;while(el.tagName!='TABLE'&&el.firstChild){el=el.firstChild;}}while(el&&el.tagName!='TABLE');return el;},unitify:function(n){if(/^\d+$/.test(n)){return n+"px";}else{return n;}},process_table:function($table){var p=$table.data('jqtp_process');if(p!==undefined){if(p.id!==undefined){$table.id=p.id;}
if(p.summary!==undefined){$table.summary=p.summary;}
if(p.caption!==undefined){$table.append("<caption>"+p.caption+"</caption>");}}
jqtp.simplify_head_and_foot($table,p);jqtp.process_rowspans($table);if(p!==undefined){jqtp.process_colours($table,p);jqtp.add_borders($table,p);jqtp.adjust_layout($table,p);jqtp.align($table,p);if(p.sort=="on"&&(p.disableallsort!="on")){$table.addClass("jqtp_sortable");}}},process_rowspans:function($t){var span=/^\s*\^\s*$/;$t.find("tr").each(function(){$(this).find("td,th")
.filter(function(){return!$(this).hasClass("jqtpRowspanned")&&span.test($(this).text());})
.each(function(){var offset=$(this).prevAll().length,rb=$(this);do{rb=rb.parent().prev()
.children().eq(offset);}while(rb.hasClass("jqtpRowspanned"));if(rb.attr("rowspan")===undefined){rb.attr("rowspan",1);}
rb.attr("rowspan",parseInt(rb.attr("rowspan"),10)+1);$(this).addClass("jqtpRowspanned");});});$t.find(".jqtpRowspanned").remove();},simplify_head_and_foot:function($table,p){if(!p)p={};var hrc=p.headerrows,frc=p.footerrows;var $tbody,$thead,thcount,$children,headrows,$kid,tdcount,footrows,$tfoot;$tbody=$table.children('tbody');if($tbody.length===0){$tbody=$('<tbody></tbody>');$thead=$table.children('THEAD');if(thead.length>0){$tbody.insertAfter($thead.first());}else{$table.prepend($tbody);}
$tbody.append($table.children('TR').remove());}
else{$tbody=$tbody.first();}
if($table.children('thead').length===0){thcount=0;$children=$tbody.children('TR');headrows=[];if(typeof(hrc)==='undefined'){hrc=$children.length;}
while(thcount<hrc){$kid=$($children[thcount]);if(!$kid.children().first().is('TH')){break;}
headrows.push($kid);thcount++;}
if(hrc>thcount){hrc=thcount;}
if(hrc>0){$thead=$("<thead></thead>");while(hrc--){$thead.append(headrows.shift().remove());}
$table.prepend($thead);}}
$table.children('tfoot')
.filter(function(){return($(this).children().length===0)})
.remove();if(frc!==undefined&&$table.children('tfoot').length==0){tdcount=0;$children=$tbody.children('TR');footrows=[];if(frc>$children.length){frc=$children.length;}
while(tdcount<frc){$kid=$($children[$children.length-1-tdcount]);if(!$kid.children().first().is('TD,TH')){break;}
footrows.push($kid);tdcount++;}
if(tdcount>0){$tfoot=$("<tfoot></tfoot>");while(frc--){$tfoot.append(footrows.pop().remove());}
$table.append($tfoot);}}},process_colours:function($t,p){var h,c,i;if(p.headerbg!==undefined||p.headercolor!==undefined){h=$t.find('thead').add($t.find('tfoot'));if(h.length){if(p.headerbg!==undefined){h.css("background-color",p.headerbg);}
if(p.headercolor!==undefined){h.css("color",p.headercolor);}}}
if(p.databg!==undefined||p.datacolor!==undefined){h=$t.find('tbody > tr');if(h.length){if(p.databg!==undefined){c=p.databg.split(/\s*,\s*/);for(i=0;i<h.length;i++){$(h[i]).css("background-color",c[i%c.length]);}}
if(p.datacolor!==undefined){c=p.datacolor.split(/\s*,\s*/);for(i=0;i<h.length;i++){$(h[i]).css("color",c[i%c.length]);}}}}},add_borders:function($t,p){if(p.tableborder!==undefined){$t[0].border=p.tableborder;}
if(p.tableframe!==undefined&&jqtp.tableframe[p.tableframe]!==undefined){$t.css('border-style',jqtp.tableframe[p.tableframe]);}
if(p.tablerules===undefined){p.tablerules="rows";}
$t[0].rules=p.tablerules;if(p.cellborder!==undefined){$t.find("td").add($t.find("th"))
.css("border-width",jqtp.unitify(p.cellborder));}},adjust_layout:function($t,p){var h,cw;if(p.cellpadding!==undefined){$t[0].cellPadding=p.cellpadding;}
if(p.cellpadding!==undefined){$t[0].cellSpacing=p.cellspacing;}
if(p.tablewidth!==undefined){$t[0].width=p.tablewidth;}
if(p.columnwidths!==undefined){cw=p.columnwidths.split(/\s*,\s*/);h=$t.find('tr').each(function(){var i=0,kid=this.firstChild,cs;while(kid&&i<cw.length){cs=kid.colSpan;if(cs<1){cs=1;}
if(cs==1){$(kid).css("width",jqtp.unitify(cw[i]));}
i+=cs;do{kid=kid.nextSibling;}while(kid&&kid.nodeType!=1);}});}},align:function($t,p){if(p.valign===undefined)
p.valign="top";if(p.headervalign===undefined){p.headervalign=p.valign;}
if(p.datavalign===undefined){p.datavalign=p.valign;}
if(p.headeralign!==undefined){$t.find("thead > tr > th")
.add($t.find("thead > tr > td"))
.add($t.find("tfoot > tr > th"))
.add($t.find("tfoot > tr > td"))
.css("vertical-align",p.headervalign)
.css("text-align",p.headeralign);}
if(p.dataalign!==undefined){$t.find("tbody > tr > td")
.add($t.find("tbody > tr > th"))
.css("vertical-align",p.datavalign)
.css("text-align",p.dataalign);}},make_sortable:function(elem){var sortOpts=$.extend({},jqtp.defaultOpts),$elem=$(elem),p=$elem.data(),sortcol=[0,0],className,cols,col;if(p.initSort!==undefined){sortcol[0]=p.initSort-1;sortOpts.sortList=[sortcol];}
if(p.initdirection!==undefined){sortcol[1]=(p.initdirection=="down")?1:0;sortOpts.sortList=[sortcol];}
if(p.databgsorted!==undefined){className='jqtp_databgsorted_'+
p.databgsorted.replace(/\W/g,'_');cols=p.databgsorted.split(/\s*,\s*/);col=cols[0];$("body").append('<style type="text/css">.'+className+
'{background-color:'+col+
'}</style>');sortOpts.cssAsc=className;sortOpts.cssDesc=className;}
if(!$elem.find("thead").length){jqtp.simplify_head_and_foot($elem);}
$elem.tablesorter(sortOpts);}};$.tablesorter.addWidget({id:'colorize',format:function(table){$(".sorted",table).removeClass("sorted");$("th.headerSortDown, th.headerSortUp",table).each(function(){var index=this.cellIndex+1;$("td:nth-child("+index+")",table).addClass("sorted");});}});$.tablesorter.addParser({id:"date",is:function(s){return!isNaN((new Date(s)).getTime());},format:function(s){return $.tablesorter.formatFloat(new Date(s).getTime());},type:"numeric"});$(function(){$(".jqtp_process").livequery(function(){jqtp.move_table_data(this);});$('table.foswikiTable').livequery(function(){jqtp.process_table($(this));});var selector=".jqtp_sortable",sort=foswiki.getPreference("JQTablePlugin.sort");if(sort){if(sort=='all'){selector+=", .foswikiTable:not(.foswikiTableInited)";}else if(sort=='attachments'){selector+=", .foswikiAttachments:not(.foswikiAttachmentsInited) table";}}
$(selector).livequery(function(){$(this).addClass("jqtp_sortable");jqtp.make_sortable(this);});});})(jQuery);
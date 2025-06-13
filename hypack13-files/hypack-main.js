

// ###########################################################
// Functions for processing the hypack menu.
// ##########################################################
function hypackShowSubMenuDelay(hypackSubMenu, hypackMenuReset) {
	document.getElementById('menuval').value = hypackSubMenu;
	setTimeout("hypackShowSubMenu(" + hypackSubMenu + ", " + hypackMenuReset + ")",100);
}


function hypackHideSubMenuDelay(hypackSubMenu, hypackMenuReset) {
	document.getElementById('menuval').value = '-1';
	setTimeout("hypackHideSubMenu(" + hypackSubMenu + ", " + hypackMenuReset + ")",100);
}


function hypackShowSubMenu(hypackSubMenu, hypackMenuReset) {
	if (document.getElementById('menuval').value == hypackSubMenu) {
		smvalue    = 'submenutab' + hypackSubMenu;
		if (document.getElementById(smvalue)) {
			document.getElementById(smvalue).style.visibility = 'visible';
		}
		mmvalue    = 'mainmenurow' + hypackSubMenu;
		tempObject = document.getElementById(mmvalue);
		tempObject.className = 'menu1slct';
	}
}
function hypackHideSubMenu(hypackSubMenu, hypackMenuReset) {
	if (document.getElementById('menuval').value != hypackSubMenu) {
		smvalue    = 'submenutab' + hypackSubMenu;
		if (document.getElementById(smvalue)) {
			document.getElementById(smvalue).style.visibility = 'hidden';
		}
		if (hypackMenuReset) {
			mmvalue    = 'mainmenurow' + hypackSubMenu;
			tempObject = document.getElementById(mmvalue);
			tempObject.className = 'menu1';
		}
	}
}

// ###########################################################
// Ticker logic for hypack.
// ##########################################################
ua=navigator.userAgent.toLowerCase();
docEl=document.getElementById?1:0;
IE=document.all&&!window.innerWidth&&ua.indexOf("msie")!=-1?1:0;
OPR=ua.indexOf("opera")!=-1&&document.clear?1:0;
OPR7=OPR&&document.appendChild?1:0;
NS=docEl&&!document.all&&ua.indexOf("opera")==-1?1:0;
NS4=document.layers?1:0;
vKON=ua.indexOf("konqueror")!=-1?parseFloat(ua.substring(ua.indexOf("konqueror/")+10)):0;
KON=vKON>=2.2?1:0;
KON3p=vKON>=3?1:0;
KON31p=vKON>=3.1?1:0;
MAC=ua.indexOf("mac")!=-1?1:0;
SAF=MAC&&ua.indexOf("safari")!=-1?1:0;
IE5M=MAC&&IE&&docEl?1:0;
IE4M=MAC&&IE&&!docEl?1:0;
IE4=!MAC&&IE&&!docEl?1:0;

tkText	= [] // array of ticker items
tkdelay	= 0;
tks2		= "";
tkwidth	= 0;
tkStartLeft	= 775;
tkCurrLeft	= tkStartLeft;
tkbuffer	= "&nbsp; &nbsp;|&nbsp; &nbsp;";

function tkscroll() {
	if (tkCurrLeft < (tkwidth * -1.5)) {
		tkCurrLeft	= tkStartLeft;
	}
	tkCurrLeft	= tkCurrLeft - 2;
	document.getElementById('tkInner').style.marginLeft = String(tkCurrLeft) + 'px';
//	document.getElementById('temp').value = tkCurrLeft;
	setTimeout("tkscroll()",30);
}

function initTicker() {

	for(i=0; i<tkText.length; i++) {
		tks2+=tkText[i]+(i==(tkText.length-1)?"":tkbuffer);
		tkwidth+=((tkText[i].length*4)+80);
	}

	tks="";
	tks += '<div id="tkHolder" style="position:relative; width:775px; height:18px; overflow:hidden;">'
	     + '<div id="tkInner" style="position:relative; margin-left:775px; padding-top:3px;">'
 	     + '<nobr>'+tks2+'</nobr>'
	     + '</div></div>';
	document.write(tks);
	setTimeout("tkscroll()",tkdelay);
}

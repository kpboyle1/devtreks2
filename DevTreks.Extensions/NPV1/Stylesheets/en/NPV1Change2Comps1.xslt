﻿<?xml version="1.0" encoding="UTF-8" ?>
<!-- Author: www.devtreks.org, 2013, December -->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
	xmlns:y0="urn:DevTreks-support-schemas:Component"
	xmlns:DisplayComps="urn:displaycomps">
	<xsl:output method="xml" indent="yes" omit-xml-declaration="yes" encoding="UTF-8" />
	<!-- pass in params -->
	<!-- what action is being taken by the server -->
	<xsl:param name="serverActionType" />
	<!-- what other action is being taken by the server -->
	<xsl:param name="serverSubActionType" />
	<!-- is the member viewing this uri the owner? -->
	<xsl:param name="isURIOwningClub" />
	<!-- which node to start with? -->
	<xsl:param name="nodeName" />
	<!-- which view to use? -->
	<xsl:param name="viewEditType" />
	<!-- is this a coordinator? -->
	<xsl:param name="memberRole" />
	<!-- what is the current uri? -->
	<xsl:param name="selectedFileURIPattern" />
	<!-- the addin being used -->
	<xsl:param name="calcDocURI" />
	<!-- the node being calculated-->
	<xsl:param name="docToCalcNodeName" />
	<!-- standard params used with calcs and custom docs -->
	<xsl:param name="calcParams" />
	<!-- what is the name of the node to be selected? -->
	<xsl:param name="selectionsNodeNeededName" />
	<!-- which network is this doc from? -->
	<xsl:param name="networkId" />
	<!-- what is the start row? -->
	<xsl:param name="startRow" />
	<!-- what is the end row? -->
	<xsl:param name="endRow" />
	<!-- what are the pagination properties ? -->
	<xsl:param name="pageParams" />
	<!-- what is the guide's email? -->
	<xsl:param name="clubEmail" />
	<!-- what is the current serviceid? -->
	<xsl:param name="contenturipattern" />
	<!-- path to resources -->
	<xsl:param name="fullFilePath" />
	<!-- init html -->
	<xsl:template match="@*|/|node()" />
	<xsl:template match="/">
		<xsl:apply-templates select="root" />
	</xsl:template>
	<xsl:template match="root">
		<div id="modEdits_divEditsDoc">
			<xsl:apply-templates select="servicebase" />
			<xsl:apply-templates select="componentgroup" />
				<div>
					<a id="aFeedback" name="Feedback">
						<xsl:attribute name="href">mailto:<xsl:value-of select="$clubEmail" />?subject=<xsl:value-of select="$selectedFileURIPattern" /></xsl:attribute>
						Feedback About <xsl:value-of select="$selectedFileURIPattern" />
					</a>
        </div>
		</div>
	</xsl:template>
	<xsl:template match="servicebase">
		<h4 class="ui-bar-b">
			Service: <xsl:value-of select="@Name" />
		</h4>
		<xsl:apply-templates select="componentgroup">
			<xsl:sort select="@Name"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="componentgroup">
		<h4>
      Component Group: <xsl:value-of select="@Name" /> ; <xsl:value-of select="@Num_0" />
    </h4>
    <xsl:apply-templates select="component">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="component">
    <xsl:variable name="fullcolcount" select="@Files + 1"/>
    <h4 class="ui-bar-b">
      Component Name, Date, Label
    </h4>
		<xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
		<xsl:for-each select="@*">
			<xsl:variable name="att_name" select="name()"/>
			<xsl:variable name="att_value" select="."/>
			<xsl:value-of select="DisplayComps:printValue('Name_', $att_name, $att_value)"/> 
		</xsl:for-each>
    <div class="ui-grid-a">
			<xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
    </div>
    <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
		<xsl:for-each select="@*">
			<xsl:variable name="att_name" select="name()"/>
			<xsl:variable name="att_value" select="."/>
			<xsl:value-of select="DisplayComps:printValue('Date_', $att_name, $att_value)"/> 
		</xsl:for-each>
    <div class="ui-grid-a">
			<xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
    </div>
    <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
		<xsl:for-each select="@*">
			<xsl:variable name="att_name" select="name()"/>
			<xsl:variable name="att_value" select="."/>
			<xsl:value-of select="DisplayComps:printValue('Num_', $att_name, $att_value)"/> 
		</xsl:for-each>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='npvchangeyr' 
      or @AnalyzerType='npvchangeid' or @AnalyzerType='npvchangealt']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="root/linkedview">
		<xsl:param name="localName" />
    <xsl:variable name="fullcolcount" select="@Files + 1"/>
      <div data-role="collapsible" data-collapsed="true" data-theme="b" data-content-theme="d" >
        <xsl:if test="($localName = 'componentgroup')">
          <h4 class="ui-bar-b">
              Component Group Details
          </h4>
        </xsl:if>
        <xsl:if test="($localName = 'component')">
          <h4 class="ui-bar-b">
            Component Details
          </h4>
        </xsl:if>
        <h4 class="ui-bar-b">
          Cost Observations 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:if test="starts-with($att_name, 'Observations_')">
					  <xsl:value-of select="DisplayComps:printValue('Observations_', $att_name, $att_value)"/> 
				  </xsl:if>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Alternative
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:if test="starts-with($att_name, 'AlternativeType_')">
					  <xsl:value-of select="DisplayComps:printValue('AlternativeType_', $att_name, $att_value)"/> 
				  </xsl:if>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          OC Total 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMOC_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          OC AmountChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMOCAmountChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          OC PercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMOCPercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          OC BaseChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMOCBaseChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          OC BasePercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMOCBasePercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          AOH Total
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMAOH_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          AOH AmountChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMAOHAmountChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          AOH PercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMAOHPercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          AOH BaseChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMAOHBaseChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          AOH BasePercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMAOHBasePercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          CAP Total 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMCAP_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          CAP AmountChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMCAPAmountChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          CAP PercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMCAPPercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          CAP BaseChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMCAPBaseChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          CAP BasePercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMCAPBasePercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Total Total 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMTOTAL_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Total AmountChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMAmountChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Total PercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMPercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Total BaseChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMBaseChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Total BasePercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMBasePercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Incent Total 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMINCENT_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Incent AmountChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMIncentAmountChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Incent PercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMIncentPercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Incent BaseChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMIncentBaseChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
        <h4 class="ui-bar-b">
          Incent BasePercentChange 
        </h4>
			  <xsl:value-of select="DisplayComps:initValues($fullcolcount)"/>
			  <xsl:for-each select="@*">
				  <xsl:variable name="att_name" select="name()"/>
				  <xsl:variable name="att_value" select="."/>
				  <xsl:value-of select="DisplayComps:printValue('TAMIncentBasePercentChange_', $att_name, $att_value)"/>
			  </xsl:for-each>
        <div class="ui-grid-a">
			    <xsl:value-of select="DisplayComps:DoPrintMobileValues($fullcolcount)"/>
        </div>
      </div>
	</xsl:template>
</xsl:stylesheet>

<div class="box" xmlns:DisplayDevPacks="urn:displaydevpacks">
  <div id="stepsmenu"><div data-role="navbar"><ul><li><a data-inline="true" data-mini="true" href="#" id="stepzero">Intro</a></li><li><a data-inline="true" data-mini="true" href="#" id="stepone">1</a></li><li><a data-inline="true" data-mini="true" href="#" id="steptwo">2</a></li><li><a data-inline="true" data-mini="true" href="#" id="stepthree">3</a></li></ul></div><div data-role="navbar"><ul><li><a data-inline="true" data-mini="true" href="#" id="stepfour">4</a></li><li><a data-inline="true" data-mini="true" href="#" id="steplast">Help</a></li></ul></div></div>
  <div id="divstepzero">
    <h4 class="ui-bar-b">
      <strong>Machinery Calculation View</strong>
    </h4>
    <h4 class="ui-bar-b">
      <strong>Machinery Calculations</strong>
    </h4>
    <p>
      <strong>Introduction</strong>
      <br />
					This tool calculates totals for input machinery uris. The discounted totals include operating,
					allocated overhead, and capital costs. Operating costs include fuel, repair and maintenance, and labor costs. 
					Allocated overhead costs include capital recovery, and taxes, housing and insurance costs. When added to 
          an operation or component, power inputs set maximum PTO horsepower properties while nonpower inputs 
          set equivalent PTO horsepower and field capacity properties.
			</p>
    <p>
      <strong>Calculation View Description</strong>
      <br />Documentation for this data can be found in a DevTreks Machinery Costs tutorial.</p>
    <p>
      <strong>Version: </strong>1.6.3</p>
    <p>
      <a id="aFeedback" name="Feedback" href="mailto:devtrekkers@devtreks.gmail?subject=crops/input/2003 Tractor, 2-Wheel Drive, 70-89 PTO HP/2147375200/none">
					Feedback About crops/input/2003 Tractor, 2-Wheel Drive, 70-89 PTO HP/2147375200/none</a>
    </p>
  </div>
  <div id="divstepone">
    <h4 class="ui-bar-b">
      <strong>Step 1 of 4. Make Selections</strong>
    </h4>
    <br /><label for="lstMachineryConstants">Machinery Constant</label><select class="" data-mini="true" id="SelectNewView855" name="crops/linkedview/Machinery Calculations/855/none;MachConstantId;integer;4"><option value="0">none</option><option value="-2094284973">Rectangular baler</option><option value="-1928212006">Side delivery rake</option><option value="-1927415477">Potato harvester</option><option value="-1452552625">Large round baler</option><option value="-1449677427">Beet topper/stalk chopper</option><option value="-1420737856">Windrower (Self Propelled)</option><option value="-1031397684">Mower-conditioner (rotary)</option><option value="-912089471">Moldboard plow</option><option value="-888784925">Mower-conditioner</option><option value="-671206352">Cotton Picker (Self Propelled)</option><option value="-652156059">Mower (rotary)</option><option value="-594121701">Mower</option><option value="-520059076">Bean puller-windrower</option><option value="-481030967">Combine (Self Propelled)</option><option value="-334610972">Forage harvester</option><option value="-274401826">Boom-type sprayer</option><option value="-265632737">Combine</option><option value="-161174521">Air-carrier sprayer</option><option value="-15039736">Wagon</option><option value="1434243">Forage wagon</option><option value="19323885">Corn picker sheller</option><option value="255933377">Grain drill</option><option value="269337832">Row crop planter</option><option value="878270992">Rotary tiller</option><option value="895936238">Row crop cultivator</option><option value="899225158">Large rectangular baler</option><option value="933992106">Rotary hoe</option><option value="1064046994">Mulcher-packer</option><option value="1212695211">Roller-packer</option><option value="1252576028">Spring tooth harrow</option><option value="1327216917">Forage harvester (Self Propelled)</option><option value="1350062964">Field cultivator</option><option value="1465169602">(Coulter) chisel plow</option><option value="1748939229">Tandem disk harrow</option><option value="1762404181">Heavy-duty disk</option><option value="1818781362">Sugar beet harvester</option><option value="1934926265">Tractor, 4 wheel drive and crawler</option><option value="1989299313">Fertilizer spreader</option><option value="2010525642">Forage blower</option><option selected="selected" value="2122503510">Tractor, 2 wheel drive and stationary tractor</option></select><p>
				  Examples:
				  <br />
					Field Efficiency: 99.000 
					Speed: 20.00 
					Width: 10.00</p><label for="lstUnitGroups">Unit</label><select class="" data-mini="true" id="SelectNewView855" name="crops/linkedview/Machinery Calculations/855/none;UnitGroupId;integer;4"><option value="0">None</option><option value="1">Metric</option><option selected="selected" value="1001">USA Standard</option></select><label for="lstCurrencyGroups">Currency</label><select class="" data-mini="true" id="SelectNewView855" name="crops/linkedview/Machinery Calculations/855/none;CurrencyGroupId;integer;4"><option value="0">none</option><option selected="selected" value="1">Dollars, US</option><option value="2">Euro, European Union</option><option value="3">Dollars, Canadian</option><option value="4">Pesos, Mexican</option></select><label for="lstRealRates">Real Rates</label><select class="" data-mini="true" id="SelectNewView855" name="crops/linkedview/Machinery Calculations/855/none;RealRateId;integer;4"><option value="0">none</option><option value="1">0.5 percent</option><option value="2">1.0 percent</option><option value="3">1.5 percent</option><option value="4">2.0 percent</option><option value="5">2.5 percent</option><option value="6">3.0 percent</option><option selected="selected" value="7">3.5 percent</option><option value="8">4.0 percent</option><option value="9">4.5 percent</option><option value="10">5.0 percent</option><option value="11">5.5 percent</option><option value="12">6.0 percent</option><option value="13">6.5 percent</option><option value="14">7.0 percent</option><option value="15">7.5 percent</option><option value="16">8.0 percent</option><option value="17">8.5 percent</option><option value="18">9.0 percent</option><option value="19">9.5 percent</option><option value="20">10.0 percent</option><option value="21">10.5 percent</option><option value="22">11.0 percent</option><option value="23">11.5 percent</option><option value="24">12.0 percent</option><option value="25">12.5 percent</option><option value="26">13.0 percent</option><option value="27">13.5 percent</option><option value="28">14.0 percent</option><option value="29">14.5 percent</option><option value="30">15.0 percent</option></select><label for="lstNominalRates">Nominal Rates</label><select class="" data-mini="true" id="SelectNewView855" name="crops/linkedview/Machinery Calculations/855/none;NominalRateId;integer;4"><option value="0">none</option><option value="1">0.5 percent</option><option value="2">1.0 percent</option><option value="3">1.5 percent</option><option value="4">2.0 percent</option><option value="5">2.5 percent</option><option value="6">3.0 percent</option><option value="7">3.5 percent</option><option value="8">4.0 percent</option><option value="9">4.5 percent</option><option value="10">5.0 percent</option><option selected="selected" value="11">5.5 percent</option><option value="12">6.0 percent</option><option value="13">6.5 percent</option><option value="14">7.0 percent</option><option value="15">7.5 percent</option><option value="16">8.0 percent</option><option value="17">8.5 percent</option><option value="18">9.0 percent</option><option value="19">9.5 percent</option><option value="20">10.0 percent</option><option value="21">10.5 percent</option><option value="22">11.0 percent</option><option value="23">11.5 percent</option><option value="24">12.0 percent</option><option value="25">12.5 percent</option><option value="26">13.0 percent</option><option value="27">13.5 percent</option><option value="28">14.0 percent</option><option value="29">14.5 percent</option><option value="30">15.0 percent</option><option value="31">15.5 percent</option><option value="32">16.0 percent</option><option value="33">16.5 percent</option><option value="34">17.0 percent</option><option value="35">17.5 percent</option><option value="36">18.0 percent</option><option value="37">18.5 percent</option><option value="38">19.0 percent</option><option value="39">19.5 percent</option><option value="40">20.0 percent</option></select><label for="lstRatingGroups">Rating Groups</label><select class="" data-mini="true" id="SelectNewView855" name="crops/linkedview/Machinery Calculations/855/none;RatingGroupId;integer;4"><option value="0">none</option><option value="1">fair quality</option><option value="2">poor quality</option><option selected="selected" value="3">good quality</option><option value="4">excellent quality</option></select></div>
  <div id="divsteptwo">
    <h4 class="ui-bar-b">
      <strong>Step 2 of 4. Make Selections</strong>
    </h4>
    <br />
    <div data-role="collapsible" data-theme="b" data-content-theme="d" data-mini="true">
      <h4 class="ui-bar-b">
        <strong>Prices</strong>
      </h4><label for="lstPriceGas">Price Gas (gallon)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceGas;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option selected="selected" value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstPriceDiesel">Price Diesel (gallon)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceDiesel;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option selected="selected" value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstPriceOil">Price Oil (gallon)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceOil;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option selected="selected" value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstPriceLP">Price LP (gallon)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceLP;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option selected="selected" value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstPriceNG">Price NG (mcf)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceNG;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option selected="selected" value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstPriceElectric">Price Electric (kwh)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceElectric;double;8"><option value="0.00">0.00</option><option value="0.01">0.01</option><option value="0.02">0.02</option><option value="0.03">0.03</option><option value="0.04">0.04</option><option value="0.05">0.05</option><option value="0.06">0.06</option><option value="0.07">0.07</option><option selected="selected" value="0.08">0.08</option><option value="0.09">0.09</option><option value="0.10">0.10</option><option value="0.11">0.11</option><option value="0.12">0.12</option><option value="0.13">0.13</option><option value="0.14">0.14</option><option value="0.15">0.15</option><option value="0.16">0.16</option><option value="0.17">0.17</option><option value="0.18">0.18</option><option value="0.19">0.19</option><option value="0.20">0.20</option><option value="0.21">0.21</option><option value="0.22">0.22</option><option value="0.23">0.23</option><option value="0.24">0.24</option></select><label for="lstPriceRegularLabor">Price Regular Labor (hour)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceRegularLabor;double;8"><option value="0.00">0.00</option><option value="1.00">1.00</option><option value="2.00">2.00</option><option value="3.00">3.00</option><option value="4.00">4.00</option><option value="5.00">5.00</option><option selected="selected" value="6.00">6.00</option><option value="7.00">7.00</option><option value="8.00">8.00</option><option value="9.00">9.00</option><option value="10.00">10.00</option><option value="11.00">11.00</option><option value="12.00">12.00</option><option value="13.00">13.00</option><option value="14.00">14.00</option><option value="15.00">15.00</option><option value="16.00">16.00</option><option value="17.00">17.00</option><option value="18.00">18.00</option><option value="19.00">19.00</option><option value="20.00">20.00</option><option value="21.00">21.00</option><option value="22.00">22.00</option><option value="23.00">23.00</option><option value="24.00">24.00</option></select><label for="lstPriceMachineryLabor">Price Machinery Labor (hour)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceMachineryLabor;double;8"><option value="0.00">0.00</option><option value="1.00">1.00</option><option value="2.00">2.00</option><option value="3.00">3.00</option><option value="4.00">4.00</option><option value="5.00">5.00</option><option value="6.00">6.00</option><option value="7.00">7.00</option><option value="8.00">8.00</option><option value="9.00">9.00</option><option value="10.00">10.00</option><option value="11.00">11.00</option><option selected="selected" value="12.00">12.00</option><option value="13.00">13.00</option><option value="14.00">14.00</option><option value="15.00">15.00</option><option value="16.00">16.00</option><option value="17.00">17.00</option><option value="18.00">18.00</option><option value="19.00">19.00</option><option value="20.00">20.00</option><option value="21.00">21.00</option><option value="22.00">22.00</option><option value="23.00">23.00</option><option value="24.00">24.00</option></select><label for="lstPriceSupervisorLabor">Price Supervisor Labor (hour)</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;PriceSupervisorLabor;double;8"><option value="0.00">0.00</option><option value="1.00">1.00</option><option value="2.00">2.00</option><option value="3.00">3.00</option><option value="4.00">4.00</option><option value="5.00">5.00</option><option value="6.00">6.00</option><option value="7.00">7.00</option><option value="8.00">8.00</option><option value="9.00">9.00</option><option value="10.00">10.00</option><option value="11.00">11.00</option><option value="12.00">12.00</option><option value="13.00">13.00</option><option value="14.00">14.00</option><option value="15.00">15.00</option><option value="16.00">16.00</option><option value="17.00">17.00</option><option selected="selected" value="18.00">18.00</option><option value="19.00">19.00</option><option value="20.00">20.00</option><option value="21.00">21.00</option><option value="22.00">22.00</option><option value="23.00">23.00</option><option value="24.00">24.00</option></select><label for="lstTaxPercent">Tax Percent (1000 (7.5% per $1000))</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;TaxPercent;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option selected="selected" value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstInsurePercent">Insure Percent (1000 (7.5% per $1000))</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;InsurePercent;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option selected="selected" value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select><label for="lstHousingPercent">Housing Percent (1000 (7.5% per $1000))</label><select class="" data-mini="true" id="SelectNewPriceView" name="crops/linkedview/Machinery Calculations/855/none;HousingPercent;double;8"><option value="0.00">0.00</option><option value="0.50">0.50</option><option value="1.00">1.00</option><option value="1.50">1.50</option><option value="2.00">2.00</option><option value="2.50">2.50</option><option value="3.00">3.00</option><option value="3.50">3.50</option><option value="4.00">4.00</option><option value="4.50">4.50</option><option selected="selected" value="5.00">5.00</option><option value="5.50">5.50</option><option value="6.00">6.00</option><option value="6.50">6.50</option><option value="7.00">7.00</option><option value="7.50">7.50</option><option value="8.00">8.00</option><option value="8.50">8.50</option><option value="9.00">9.00</option><option value="9.50">9.50</option><option value="10.00">10.00</option><option value="10.50">10.50</option><option value="11.00">11.00</option><option value="11.50">11.50</option><option value="12.00">12.00</option></select></div>
    <div data-role="collapsible" data-theme="b" data-content-theme="d" data-mini="true">
      <h4 class="ui-bar-b">
        <strong>Optional Machinery Selection and Scheduling Size Range</strong>
      </h4>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="size1">Size (Width) 1 </label></div>
        <div class="ui-block-b">
          <label for="lp1">List Price 1 </label></div>
        <div class="ui-block-a">
          <label for="max1">HP (Max PTO HP) 1 </label></div>
        <div class="ui-block-b">
          <label for="p1">Speed 1 </label></div>
        <div class="ui-block-a">
          <label for="fe1">Field Efficiency 1 </label></div>
        <div class="ui-block-b">
          <label for="epto1">Equiv PTO HP 1 </label></div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="size2">Size (Width) 2 </label></div>
        <div class="ui-block-b">
          <label for="lp2">List Price 2 </label></div>
        <div class="ui-block-a">
          <label for="max2">HP (Max PTO HP) 2 </label></div>
        <div class="ui-block-b">
          <label for="p2">Speed 2 </label></div>
        <div class="ui-block-a">
          <label for="fe2">Field Efficiency 2 </label></div>
        <div class="ui-block-b">
          <label for="epto2">Equiv PTO HP 2 </label></div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="size3">Size (Width) 3 </label></div>
        <div class="ui-block-b">
          <label for="lp3">List Price 3 </label></div>
        <div class="ui-block-a">
          <label for="max3">HP (Max PTO HP) 3 </label></div>
        <div class="ui-block-b">
          <label for="p3">Speed 3 </label></div>
        <div class="ui-block-a">
          <label for="fe3">Field Efficiency 3 </label></div>
        <div class="ui-block-b">
          <label for="epto3">Equiv PTO HP 3 </label></div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="size4">Size (Width) 4 </label></div>
        <div class="ui-block-b">
          <label for="lp4">List Price 4 </label></div>
        <div class="ui-block-a">
          <label for="max4">HP (Max PTO HP) 4 </label></div>
        <div class="ui-block-b">
          <label for="p4">Speed 4 </label></div>
        <div class="ui-block-a">
          <label for="fe4">Field Efficiency 4 </label></div>
        <div class="ui-block-b">
          <label for="epto4">Equiv PTO HP 4 </label></div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="size5">Size (Width) 5 </label></div>
        <div class="ui-block-b">
          <label for="lp5">List Price 5 </label></div>
        <div class="ui-block-a">
          <label for="max5">HP (Max PTO HP) 5 </label></div>
        <div class="ui-block-b">
          <label for="p5">Speed 5 </label></div>
        <div class="ui-block-a">
          <label for="fe5">Field Efficiency 5 </label></div>
        <div class="ui-block-b">
          <label for="epto5">Equiv PTO HP 5 </label></div>
      </div>
    </div>
  </div>
  <div id="divstepthree">
    <h4 class="ui-bar-b">
      <strong>Step 3 of 4. Calculate</strong>
    </h4>
    <div data-role="collapsible" data-theme="b" data-content-theme="d">
      <h4>Relations</h4><div data-role="fieldcontain"><fieldset data-mini="true" data-role="controlgroup" data-type="horizontal"><legend>Use In Childs?</legend><input checked="checked" data-mini="true" id="lblUseSame1" name="" type="radio" value="1" /><label for="lblUseSame1">True</label><input data-mini="true" id="lblUseSame2" name="" type="radio" value="0" /><label for="lblUseSame2">False</label></fieldset></div><div data-role="fieldcontain"><fieldset data-mini="true" data-role="controlgroup" data-type="horizontal"><legend>Overwrite Childs?</legend><input checked="checked" data-mini="true" id="lblOverwrite1" name="" type="radio" value="1" /><label for="lblOverwrite1">True</label><input data-mini="true" id="lblOverwrite2" name="" type="radio" value="0" /><label for="lblOverwrite2">False</label></fieldset></div><div class="ui-grid-a"><div class="ui-block-a"><label for="lblWhatIfTagName">What If Tag</label><input data-mini="true" id="lblWhatIfTagName" name="" type="text" value="none"></input></div><div class="ui-block-b"><label for="lblRelatedCalculatorsType">Related C. Type</label><input data-mini="true" id="lblRelatedCalculatorsType" name="" type="text" value="agmachinery"></input></div></div></div>
    <div data-role="collapsible" data-collapsed="false" data-theme="b" data-content-theme="d" data-mini="true">
      <h4 class="ui-bar-b">
        <strong>Operating Costs</strong>
      </h4>
      <div class="ui-grid-a">
        <div class="ui-block-a">
                    Area hours/acre :
                  0.0417</div>
        <div class="ui-block-b">
                    Fuel (gal/hr):
                  7.2779</div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
                  Fuel Cost: 14.8370</div>
        <div class="ui-block-b">
                  Lube Oil Cost: 0.0783</div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
                  Repair Cost: 4.5446</div>
        <div class="ui-block-b">
                  Labor Cost: 13.4551</div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <strong>Total Operating Cost ($/hour): </strong>
        </div>
        <div class="ui-block-b">
          <strong>32.915</strong>
        </div>
      </div>
      <h4 class="ui-bar-b">
        <strong>Allocated Overhead Costs</strong>
      </h4>
      <div class="ui-grid-a">
        <div class="ui-block-a">
                  Capital Recovery Cost: 6.6278</div>
        <div class="ui-block-b">
                  Taxes, Housing, Insurance: 0.5471</div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <strong>Total Allocated Overhead Cost ($/hour): </strong>
        </div>
        <div class="ui-block-b">
          <strong>7.175</strong>
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <strong>Capital Cost:</strong>33600.000</div>
        <div class="ui-block-b">
                  Capital Unit: each</div>
      </div>
    </div>
    <div data-role="collapsible" data-theme="b" data-content-theme="d" data-mini="true">
      <h4 class="ui-bar-b">
        <strong>A. Select options</strong>
      </h4>
      <div class="ui-field-contain">
        <fieldset data-role="controlgroup" data-type="horizontal">
          <legend>Capacity Options:</legend>
          <input type="radio" id="CapacityOptions1" data-mini="true" value="1" checked="true" />
          <label for="CapacityOptions1">Area (hours/acre)</label>
          <input type="radio" id="CapacityOptions2" data-mini="true" value="2" />
          <label for="CapacityOptions2">Material (hours/ton)</label>
          <input type="radio" id="CapacityOptions3" data-mini="true" value="3" />
          <label for="CapacityOptions3">Area (acres/hour)</label>
          <input type="radio" id="CapacityOptions4" data-mini="true" value="4" />
          <label for="CapacityOptions4">Material (tons/hour)</label>
        </fieldset>
      </div>
      <div class="ui-field-contain">
        <fieldset data-role="controlgroup" data-type="horizontal">
          <legend>Vary Time Options</legend>
          <input type="radio" id="TimeandOutputOptions1" data-mini="true" value="1" checked="true" />
          <label for="TimeandOutputOptions1">Costs Vary Over Time</label>
          <input type="radio" id="TimeandOutputOptions2" data-mini="true" value="2" />
          <label for="TimeandOutputOptions2">Costs Do Not VOT</label>
        </fieldset>
      </div>
      <div class="ui-field-contain">
        <fieldset data-role="controlgroup" data-type="horizontal">
          <legend>Inflation Options</legend>
          <input type="radio" id="InflationOptions1" data-mini="true" value="1" checked="true" />
          <label for="InflationOptions1">First Year</label>
          <input type="radio" id="InflationOptions2" data-mini="true" value="2" />
          <label for="InflationOptions2">All Years</label>
          <input type="radio" id="InflationOptions3" data-mini="true" value="3" />
          <label for="InflationOptions3">Do Not Use</label>
        </fieldset>
      </div>
      <div class="ui-field-contain">
        <fieldset data-role="controlgroup" data-type="horizontal">
          <legend>Fuel Options</legend>
          <input type="radio" id="FuelOptions1" data-mini="true" value="1" checked="true" />
          <label for="FuelOptions1">Base on Operation</label>
          <input type="radio" id="FuelOptions2" data-mini="true" value="2" />
          <label for="FuelOptions2">Base on Enterprise</label>
        </fieldset>
      </div>
    </div>
    <div data-role="collapsible" data-theme="b" data-content-theme="d" data-mini="true">
      <h4 class="ui-bar-b">
        <strong>B. Fill in machinery variables</strong>
      </h4>
      <div>
        <label for="CalculatorName" class="ui-hidden-accessible" />
        <input id="CalculatorName" type="text" data-mini="true" value="Machinery Calculations" />
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblFuelType">Fuel Type</label>
          <select class="Select225" id="lblFuelType" data-mini="true">
            <option value="none">none
								</option>
            <option value="diesel">diesel
								</option>
            <option value="gas" selected="">gas
								</option>
            <option value="lpg">lpg
								</option>
          </select>
        </div>
        <div class="ui-block-b" />
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblMarketValue">Market Value (input.CAPPrice)</label>33,600.0000</div>
        <div class="ui-block-b">
          <label for="lblPlannedUseHours">Planned Use Hours</label>
          <input id="lblPlannedUseHours" type="text" data-mini="true" value="500" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblSalvageValue">Salvage Value</label>
          <input type="text" id="lblSalvageValue" data-mini="true" value="3300.00" />
        </div>
        <div class="ui-block-b">
          <label for="lblStartingHours">Starting Hours</label>
          <input type="text" id="lblStartingHours" data-mini="true" value="6000" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblHorsePower">Horse Power</label>
          <input type="text" id="lblHorsePower" data-mini="true" value="80" />
        </div>
        <div class="ui-block-b">
          <label for="lblUsefulLifeHours">Useful Life Hours</label>
          <input type="text" id="lblUsefulLifeHours" data-mini="true" value="12000" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblHPPTOMax">Max PTO HP</label>
          <input type="text" id="lblHPPTOMax" data-mini="true" value="70" />
        </div>
        <div class="ui-block-b">
          <label for="lblServiceCapacity">Service Capacity (area covered)</label>
          <input type="text" id="lblServiceCapacity" data-mini="true" value="0.0417" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblHPPTOEquiv">Equiv PTO HP</label>
          <input type="text" id="lblHPPTOEquiv" data-mini="true" value="70" />
        </div>
        <div class="ui-block-b">
          <label for="lblFieldSpeedTypical">Field Speed Typical</label>
          <input type="text" id="lblFieldSpeedTypical" data-mini="true" value="20.0000" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblListPriceAdj">List Price Adj (+)</label>
          <input type="text" id="lblListPriceAdj" data-mini="true" value="10.0000" />
        </div>
        <div class="ui-block-b">
          <label for="lblWidth">Width</label>
          <input type="text" id="lblWidth" data-mini="true" value="10.0000" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblFieldEffTypical">Field Eff Typical</label>
          <input type="text" id="lblFieldEffTypical" data-mini="true" value="99.000" />
        </div>
        <div class="ui-block-b">
          <label for="lblDate">Date</label>
          <input type="text" id="lblDate" name="lblDate" data-mini="true" disabled="true" value="09/08/2017" />
        </div>
      </div>
      <div class="ui-grid-a">
        <div class="ui-block-a">
          <label for="lblLaborType">Labor Type</label>
          <select class="Select225" id="lblLaborType" data-mini="true">
            <option value="none">none
							</option>
            <option value="regular">regular
							</option>
            <option value="machinery" selected="">machinery
							</option>
            <option value="supervisory">supervisory
							</option>
          </select>
        </div>
        <div class="ui-block-b">
          <label for="lblLaborAmountAdj">Labor Amount Adj</label>
          <input type="text" id="lblLaborAmountAdj" data-mini="true" value="10.0000" />
        </div>
      </div>
      <div>
        <label for="lblDescription">Description</label>
        <textarea class="Text75H100PCW" id="lblDescription" data-mini="true">Documentation for this data can be found in a DevTreks Machinery Costs tutorial.</textarea>
      </div>
      <div>
        <label for="lblMediaURL">Media URL</label>
        <textarea class="Text75H100PCW" id="lblMediaURL" data-mini="true"></textarea>
      </div>
    </div>
  </div>
  <div id="divstepfour">
    <h4 class="ui-bar-b">
      <strong>Step 4 of 4. Save</strong>
    </h4>
    <p>
      <strong>Method 1.</strong> Do you wish to save step 2's calculations? These calculations are viewed by opening this particular calculator addin.
					</p>
  </div>
  <div id="divsteplast">
    <h4 class="ui-bar-b">
      <strong>Instructions (beta)</strong>
    </h4>
    <div data-role="collapsible" data-theme="b" data-content-theme="d">
      <h4>Step 1</h4>
      <ul data-role="listview">
        <li>
          <strong>Step 1. Machinery Constants:</strong> The constants derive from ASABE publication ASAE D497.7 listed below.</li>
        <li>
          <strong>Step 1. Real and Nominal Rates:</strong> The nominal rate includes inflation while the real rate does not. In the USA, DevTreks recommends 
					using Office of Management and Budget rates for the same year as the date of the input.</li>
      </ul>
    </div>
    <div data-role="collapsible" data-theme="b" data-content-theme="d">
      <h4>Step 2</h4>
      <ul data-role="listview">
        <li>
          <strong>Step 2. Fuel Prices:</strong> The prices should be as of the date of the input. Prices that won't be used, such as natural gas prices, do not need to be filled out.</li>
        <li>
          <strong>Step 2. Labor Prices:</strong> These prices should reflect actual gross hourly wages for each category of labor.</li>
        <li>
          <strong>Step 2. Tax Percent:</strong> A percent that is multiplied by the market value to derive a tax cost. Similar to mill rates (percent tax per $1000 of assessed value). </li>
        <li>
          <strong>Step 2. Housing Percent:</strong> A percent that is multiplied by the market value to derive a housing cost. </li>
        <li>
          <strong>Step 2. Insure Percent:</strong> A percent that is multiplied by the market value to derive an insurance cost. </li>
        <li>
          <strong>Step 2. Size Ranges:</strong> This information is optional. Machinery selection and scheduling analyzers use these ranges to determine feasible combinations 
          of machinery (i.e. least cost) in operations and components. The numbers entered should be reasonably related, but not equal, to the associated numbers entered in step 3. 
          The size column is width.  The HP column should be maximum PTO horsepower, or horsepower, for power inputs. For nonpower inputs, 
          the number should reflect an associated, typical, power input. The speed and efficiency columns are used with the width column to determine field, or service, capacity. 
          The Equivalent PTO HP is used to determine fuel amounts and costs. If the last three columns are left blank or zero, ratios derived from step 3 will be used in calculations.
          The references below explain how these parameters are used in machinery selection and scheduling.</li>
      </ul>
    </div>
    <div data-role="collapsible" data-theme="b" data-content-theme="d">
      <h4>Step 3</h4>
      <ul data-role="listview">
        <li>
          <strong>Step 3. Use Same Calculator Pack In Descendants?:</strong> True to insert or update this same calculator in children.</li>
        <li>
          <strong>Step 3. Overwrite Descendants?:</strong> True to insert or update all of the attributes of this same calculator in all children. False only updates children attributes that are controlled by the developer of this calculator (i.e. version, stylehsheet name, relatedcalculatorstype ...)</li>
        <li>
          <strong>Step 3. What If Tag Name:</strong> Instructional videos explaining the use of what-if scenario calculations should be viewed before changing or using this parameter.</li>
        <li>
          <strong>Step 3. Related Calculators Type:</strong> When the Use Same Calculator Pack in Descendant is true, uses this value to determine which descendant calculator to update. Inserts a new descendant when no descendant has this same name. Updates the descendant that has this same name.</li>
        <li>
          <strong>Step 3. Capacity Options:</strong> The materials option is only used when machinery, such as balers, do not cover each square foot of a field.</li>
        <li>
          <strong>Step 3. Vary Time and Output Options:</strong> These options change the the operating costs (fuel, repair, labor, lube). The 'Costs Vary Over Time' option calculates an amortized average annual repair cost adjusted for inflation (see Hallam et al equation 5.18). The 'Costs Do Not Vary Over Time' option calculates repair costs, with no inflation adjustments, using ASABE repair cost equations.</li>
        <li>
          <strong>Step 3. Inflation Options:</strong> These options change the operating and capital recovery costs.</li>
        <li>
          <strong>Step 3. Fuel Options:</strong> Machinery that is being used for specific operations, such as planting, should use the 'Operation' option. Machinery that may be used in several operations should use the 'Enterprise' option.</li>
        <li>
          <strong>Step 3. Market Value:</strong> Before a calculation is run for the first time, this number will be zero. When the calculation is run the parent input's CAPPrice is used as the market value.</li>
        <li>
          <strong>Step 3. List Price Adjustment:</strong> A multiplier that is used to increase the market value to the list price. For example, if the list price of an item of machinery is ten percent more that the market value, enter 10. This number is divided by 100 in the calculations.</li>
        <li>
          <strong>Step 3. Scheduling Efficiency and Labor Adjustment:</strong> A multiplier that is used to adjust the amount of labor needed to use the machinery for the specified area (i.e. acre or hectare). For example, if the time spent setting up the machinery or transporting the machinery adds ten percent to the total amount of time it takes to complete the specified area, enter 10. If the scheduling of the machinery's use often causes the equipment operator to be idle, add an additional adjustment. This number is divided by 100 in the calculations.</li>
        <li>
          <strong>Step 3. Field Efficiency Typical:</strong> "The ratio between the productivity of a machine under field conditions and the theoretical maximum productivity" (see ASABE). In general, we recommend using the default ASABE value that comes from selecting a machinery constant. This number is divided by 100 in the calculations. Note that all fields requiring a percent in DevTreks' calculators should be entered as a whole number (unless the percent is less than 1%).</li>
      </ul>
      <ul data-role="listview">
        <li>
          <strong>Step 3. Area:</strong> Also known as service capacity or effective field capacity, is the number of hours used by the machine per acre (hectare). This number is used to set operating and allocated overhead cost amounts.</li>
        <li>
          <strong>Step 3. Per Hour Costs:</strong> With the exception of the capital cost, all costs are per hour of machinery use.  Total cost per acre is calculated at cost (per hour) x area covered (hours per acre). For example, if an implement is used for .10 hours per acre and the repair cost is $10 per hour, the total repair cost per acre is $1 per acre. </li>
        <li>
          <strong>Step 3. Fuel Cost:</strong> The quantity of fuel consumed per hour (by the power input) multiplied by the fuel price.</li>
        <li>
          <strong>Step 3. Repair Cost:</strong> 'Includes replacement parts, materials, shop expenses, and labor for maintaining a machine in good working order.'(ASABE)</li>
        <li>
          <strong>Step 3. Labor Cost:</strong> The quantity of labor used per hour multiplied by the labor price.</li>
        <li>
          <strong>Step 3. Lube Cost:</strong> 'Oil consumption is defined as the volume per hour of engine crankcase oil replaced at the manufacturers recommended change interval' (ASABE).</li>
        <li>
          <strong>Step 3. Operating Cost:</strong> Costs associated with the resources expended (expendables) during the production cycle.</li>
        <li>
          <strong>Step 3. Capital Recovery Cost:</strong> Annual ownership cost, adjusted for inflation (unless the the no inflation option is chosen). 'This cost reflects the reduction in value of an asset with use and time.' (ASABE)</li>
        <li>
          <strong>Step 3. Taxes, Housing and Insurance Cost:</strong> These are calculated as a percent of the market value of the machinery.</li>
        <li>
          <strong>Step 3. Allocated Overhead Cost:</strong> All costs that are not operating costs (that are not expendables), such as capital costs. </li>
        <li>
          <strong>Step 3. Capital Cost:</strong> Initial market value (purchase price) of the machinery.</li>
      </ul>
    </div>
    <div data-role="collapsible" data-theme="b" data-content-theme="d">
      <h4 class="ui-bar-b">
        <strong>References</strong>
      </h4>
      <ul data-role="listview">
        <li>
          <strong>American Society of Agricultural and Biological Engineers, ASAE D497.7 MAR2011</strong> Agricultural Machinery Management Data</li>
        <li>
          <strong>American Society of Agricultural and Biological Engineers, ASAE EP496.3 FEB2006 (R2011)</strong> Agricultural Machinery Management</li>
        <li>
          <strong>Hallam, Eidman, Morehart and Klonsky (editors) </strong> Commodity Cost and Returns Estimation Handbook, Staff General Research Papers, Iowa State University, Department of Economics, 1999</li>
      </ul>
    </div>
  </div>
  <div>
    <br />
    <strong>Current view of document</strong>
  </div>
</div>